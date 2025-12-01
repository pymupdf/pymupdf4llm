
import os
import sys
import time
import pickle

import torch
import anyconfig
import copy

import cv2
import blosc
import pymupdf

import numpy as np
import multiprocessing as mp

import torch.distributed as dist

from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from train.core.common.util import print_network
from train.core.common.model_util import save_model_and_optimizer, load_model_and_optimizer
from train.core.model.ModelFactory import get_model

from train.tools.data.layout.DocumentJsonDataset import DocumentJsonDataset
from train.tools.data.layout.DocumentLMDBDataset import DocumentLMDBDataset

from focal_loss.focal_loss import FocalLoss
from train.tools.schedulers.CustomCyclicLR import CustomCyclicLR
from train.tools.schedulers.CustomStepCycleLR import CustomStepCycleLR


cfg = None

# --- F1 스코어 계산 함수 ---
def calculate_f1_scores(node_preds, node_targets, edge_preds, edge_targets):
    """
    노드 및 엣지 분류에 대한 F1 스코어를 계산합니다.
    """
    node_f1 = f1_score(node_targets.cpu().numpy(), node_preds.cpu().numpy(), average='weighted', zero_division=0)
    edge_f1 = f1_score(edge_targets.cpu().numpy(), edge_preds.cpu().numpy(), average='binary', zero_division=0)
    return node_f1, edge_f1


def evaluate_model(cfg, model, dataloader, node_criterion, edge_criterion, device):
    """
    Evaluates the model's performance on a given dataset.

    Args:
        model: The model to be evaluated.
        dataloader: DataLoader for the evaluation dataset.
        node_criterion: Loss function for node prediction. Can be None.
        edge_criterion: Loss function for edge prediction. Can be None.
        device: The device (CPU or GPU) to run the evaluation on.

    Returns:
        A tuple containing average losses and F1 scores.
    """
    model.eval()  # Set the model to evaluation mode
    val_node_preds = []
    val_node_targets = []
    val_edge_preds = []
    val_edge_targets = []
    val_total_loss = 0
    val_total_node_loss = 0
    val_total_edge_loss = 0

    with torch.no_grad():  # Disable gradient calculation for inference
        for val_batch in dataloader:
            val_batch = val_batch.to(device)
            val_node_logits, val_edge_logits = model.forward_with_batch(val_batch)

            # Calculate losses based on whether the criterion is provided
            val_node_loss = 0
            if node_criterion is not None:
                node_loss_type = cfg['train']['loss']['node_loss_type']
                if node_loss_type == 'focal':
                    soft_max = torch.nn.Softmax(dim=-1)
                    val_node_logits = soft_max(val_node_logits)
                val_node_loss = node_criterion(val_node_logits, val_batch.y)

            val_edge_loss = 0
            if edge_criterion is not None:
                edge_loss_type = cfg['train']['loss']['edge_loss_type']
                if edge_loss_type == 'focal':
                    soft_max = torch.nn.Softmax(dim=-1)
                    val_edge_logits = soft_max(val_edge_logits)
                val_edge_loss = edge_criterion(val_edge_logits, val_batch.edge_label)

            # Sum up losses
            val_loss = val_node_loss + val_edge_loss
            val_total_loss += val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
            val_total_node_loss += val_node_loss.item() if isinstance(val_node_loss, torch.Tensor) else val_node_loss
            val_total_edge_loss += val_edge_loss.item() if isinstance(val_edge_loss, torch.Tensor) else val_edge_loss

            # Store predictions for metric calculation
            val_node_preds.append(torch.argmax(val_node_logits, dim=1).cpu())
            val_node_targets.append(val_batch.y.cpu())
            val_edge_preds.append(torch.argmax(val_edge_logits, dim=1).cpu())
            val_edge_targets.append(val_batch.edge_label.cpu())

    # Concatenate all predictions and targets
    val_node_preds = torch.cat(val_node_preds)
    val_node_targets = torch.cat(val_node_targets)
    val_edge_preds = torch.cat(val_edge_preds)
    val_edge_targets = torch.cat(val_edge_targets)

    # Calculate F1 scores
    node_f1, edge_f1 = calculate_f1_scores(val_node_preds, val_node_targets,
                                           val_edge_preds, val_edge_targets)

    avg_val_loss = val_total_loss / len(dataloader)
    avg_val_node_loss = val_total_node_loss / len(dataloader)
    avg_val_edge_loss = val_total_edge_loss / len(dataloader)

    model.train()
    return avg_val_loss, avg_val_node_loss, avg_val_edge_loss, node_f1, edge_f1

def update_ema_weights(model, ema_model, decay):
    """
    Update EMA weights using the current model's weights.
    """
    with torch.no_grad():
        for ema_param, current_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.copy_(decay * ema_param + (1 - decay) * current_param)


def copy_ema_to_model(model, ema_model):
    """
    Copy EMA model's parameters to the main model.
    """
    with torch.no_grad():
        for model_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            model_param.copy_(ema_param)


def train(cfg, rank=0, world_size=1):
    is_ddp_training = world_size > 1

    device = cfg['train']['device']
    save_dir = cfg['train']['save_dir']
    load_from = cfg['train']['load_from']
    log_step = cfg['train']['log_step']
    log_clear_step = cfg['train']['log_clear_step']
    save_step = cfg['train']['save_step']
    eval_step = cfg['train']['eval_step']
    sleep_time = cfg['train']['sleep_time']

    num_epoch = cfg['train']['num_epoch']
    ema_decay = cfg['train']['ema_decay']
    gradient_clip_norm = cfg['train']['gradient_clip_norm']

    # Init DDP
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device('cpu')

    if is_ddp_training:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if rank == 0:
            print(f"Initialized DDP on Rank {rank}, using device {device}")

    auto_restart_lr = None
    auto_restart_step = None
    if cfg['train']['auto_restart']:
        auto_restart_lr = cfg['train']['scheduler']['cycle_steps']
        step = cfg['train']['scheduler']['step_size']
        if type(step) is int:
            auto_restart_step = [ step ] * len(auto_restart_lr)
        else:
            assert len(step) == len(auto_restart_lr)
            auto_restart_step = step

    train_dataset_type = cfg['train']['train_dataset']['type']
    train_lmdb_path = cfg['train']['train_dataset']['lmdb_path']
    train_lmdb_prob = cfg['train']['train_dataset']['lmdb_prob']
    train_dataset_cache_size = cfg['train']['train_dataset']['cache_size']
    train_dataset_cache_rate = cfg['train']['train_dataset']['cache_rate']

    val_dataset_type = cfg['train']['val_dataset']['type']
    val_lmdb_path = cfg['train']['val_dataset']['lmdb_path']

    train_batch_size = cfg['train']['train_batch_size']
    val_batch_size = cfg['train']['val_batch_size']

    train_loss_weight = cfg['train']['loss']['weight']

    sample_k = cfg['model']['sample_k']

    data_class_names = cfg['data']['class_list']
    data_class_map = {}
    for i in range(len(data_class_names)):
        data_class_map[data_class_names[i]] = i

    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/model.yaml', "w") as f:
        model_cfg = {
            'data': cfg['data'],
            'model': cfg['model']
        }
        anyconfig.dump(model_cfg, f)

    with open(f'{save_dir}/train.yaml', "w") as f:
        model_cfg = {
            'train': cfg['train'],
        }
        anyconfig.dump(model_cfg, f)

    data_rf_names = cfg['data']['rf_names']
    if train_dataset_type == 'json':
        train_json_path = cfg['train']['train_dataset']['json_path']
        train_data = DocumentJsonDataset(data_path=train_json_path,  k=sample_k, class_map=data_class_map, is_train=True,
                                         cache_size=train_dataset_cache_size, cache_rate=train_dataset_cache_rate)
    elif train_dataset_type == 'lmdb':
        train_data = DocumentLMDBDataset(lmdb_path=train_lmdb_path, data_prob=train_lmdb_prob, cache_size=train_dataset_cache_size,
                                         rf_names=data_rf_names)
    else:
        raise Exception(f'Invalid train_dataset_type = {train_dataset_type}')

    # For DDP training
    if is_ddp_training:
        sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle_flag = False
        train_dataloader = GeometricDataLoader(train_data, batch_size=train_batch_size,
                                               sampler=sampler, shuffle=shuffle_flag,
                                               num_workers=0, persistent_workers=False, drop_last=True)
    # For normal training
    else:
        train_dataloader = GeometricDataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                               persistent_workers=False, drop_last=True)

    if val_dataset_type == 'json':
        val_json_path = cfg['train']['val_dataset']['json_path']
        val_data = DocumentJsonDataset(data_path=val_json_path, k=sample_k, class_map=data_class_map, is_train=True)
    elif val_dataset_type == 'lmdb':
        val_data = DocumentLMDBDataset(lmdb_path=val_lmdb_path, cache_size=0,
                                       rf_names=data_rf_names)
    else:
        raise Exception(f'Invalid train_dataset_type = {train_dataset_type}')
    val_dataloader = GeometricDataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=0,
                                         persistent_workers=False)

    val_dataloader2 = None
    save_dir2 = None
    if 'val_dataset2' in cfg['train']:
        val_lmdb_path2 = cfg['train']['val_dataset2']['lmdb_path']
        save_dir2 = cfg['train']['val_dataset2']['save_dir']
        val_data2 = DocumentLMDBDataset(lmdb_path=val_lmdb_path2, cache_size=0, rf_names=data_rf_names)
        val_dataloader2 = GeometricDataLoader(val_data2, batch_size=val_batch_size, shuffle=False, num_workers=0,
                                              persistent_workers=False)
        os.makedirs(save_dir2, exist_ok=True)
        with open(f'{save_dir2}/model.yaml', "w") as f:
            anyconfig.dump(model_cfg, f)

    model_0 = get_model(cfg, data_class_names)
    model_0 = model_0.to(device)
    if is_ddp_training:
        model_0 = DDP(model_0, device_ids=[device_id], output_device=device_id)

    model_to_use = model_0.module if is_ddp_training else model_0
    print_network(model_to_use, verbose=True)

    opt_type = cfg['train']['optimizer']['type']
    lr = float(cfg['train']['optimizer']['learning_rate'])

    if opt_type == 'SGD':
        momentum = float(cfg['train']['optimizer']['momentum'])
        optimizer = torch.optim.SGD(model_0.parameters(), lr=lr, momentum=momentum)
    elif opt_type == 'Adam':
        betas = cfg['train']['optimizer']['betas']
        weight_decay = float(cfg['train']['optimizer']['weight_decay'])
        optimizer = torch.optim.Adam(model_0.parameters(), lr=lr,  weight_decay=weight_decay, betas=betas)
    else:
        raise Exception(f'Invalid optimizer type = {opt_type}!')

    if load_from is not None and os.path.exists(load_from):
        print(f"Loading model from {load_from}...")
        opt = None
        if cfg['train']['load_optimizer']:
            opt = optimizer

        # Load the state dictionary with strict=False
        load_model_and_optimizer(model_to_use, load_from, optimizer=opt)
        print("Model loaded successfully!")

    ema_model = copy.deepcopy(model_to_use)
    ema_model.eval()

    scheduler = None
    if auto_restart_lr is None:
        schd_type = cfg['train']['scheduler']['type']
        if schd_type == 'CLR':
            base_lr = float(cfg['train']['scheduler']['base_lr'])
            max_lr = float(cfg['train']['scheduler']['max_lr'])
            step_size = int(cfg['train']['scheduler']['step_size'])
            gamma = float(cfg['train']['scheduler']['gamma'])
            if gamma < 1.0:
                scheduler = CustomCyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, decay_factor=gamma)
            else:
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size,
                                                              cycle_momentum=False)
        elif schd_type == 'StepCycle':
            steps = cfg['train']['scheduler']['cycle_steps']
            step_size = int(cfg['train']['scheduler']['step_size'])
            scheduler = CustomStepCycleLR(optimizer, steps=steps, step_size=step_size)

    node_loss_type = cfg['train']['loss']['node_loss_type']
    node_weight = cfg['train']['loss']['node_weight']
    if node_weight is not None:
        node_weight = torch.tensor(node_weight, dtype=torch.float).to(device)
    if node_loss_type == 'CE':
        node_criterion = torch.nn.CrossEntropyLoss(weight=node_weight)
    elif node_loss_type == 'focal':
        node_criterion = FocalLoss(gamma=2.0)
    else:
        raise Exception(f'Invalid node_loss : {node_loss_type}')

    edge_loss_type = cfg['train']['loss']['edge_loss_type']
    edge_weight = cfg['train']['loss']['edge_weight']
    if edge_weight is not None:
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device)
    if edge_loss_type == 'CE':
        edge_criterion = torch.nn.CrossEntropyLoss(weight=edge_weight)
    elif edge_loss_type == 'focal':
        edge_criterion = FocalLoss(gamma=2.0)
    else:
        raise Exception(f'Invalid node_loss : {node_loss_type}')

    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    warmup_steps = 500
    base_lr = 1e-6
    target_lr = 1e-4

    # TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0:
        log_dir = os.path.join(save_dir, "tf_logs")
        writer = SummaryWriter(log_dir=log_dir)

    model_to_use.train()
    num_step = 0
    print("--- Training Start ---")
    # --------------------------------------------
    restart_index = 0  # current index in auto_restart_lr/step
    steps_in_phase = 0  # counter for current phase steps
    used_restart_files = set()

    if auto_restart_lr is not None and auto_restart_step is not None:
        scheduler = None
        # Set the initial LR from auto_restart config
        for g in optimizer.param_groups:
            target_lr = float(auto_restart_lr[restart_index])
        print(
            f"[Auto-Restart] Starting with LR={auto_restart_lr[restart_index]} for {auto_restart_step[restart_index]} steps")

    import time
    for epoch in range(num_epoch):
        if is_ddp_training:
            train_dataloader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_dataloader):
            num_step += 1
            steps_in_phase += 1

            if steps_in_phase < warmup_steps:
                new_lr = base_lr + (target_lr - base_lr) * (steps_in_phase / warmup_steps)
                set_lr(optimizer, new_lr)
            else:
                set_lr(optimizer, target_lr)

            start_time = time.time()
            batch = batch.to(device)
            data_load_time = time.time() - start_time

            start_time = time.time()
            node_logits, edge_logits = model_to_use.forward_with_batch(batch)

            if node_loss_type == 'focal':
                soft_max = torch.nn.Softmax(dim=-1)
                node_logits = soft_max(node_logits)
            if edge_loss_type == 'focal':
                soft_max = torch.nn.Softmax(dim=-1)
                edge_logits = soft_max(edge_logits)

            node_loss = node_criterion(node_logits, batch.y)
            edge_loss = edge_criterion(edge_logits, batch.edge_label)

            # ignore nan-loss
            if torch.isnan(node_loss).any() or torch.isnan(edge_loss).any():
                continue

            if train_loss_weight is not None:
                total_loss = (train_loss_weight[0] * node_loss) + (train_loss_weight[1] * edge_loss)
            else:
                try:
                    if is_ddp_training and rank != 0:
                        loss_weight = torch.tensor(1.0).to(device)
                    else:
                        loss_weight = (edge_loss / node_loss).clone().detach()
                    if is_ddp_training:
                        dist.broadcast(loss_weight.data, src=0)
                    loss_weight = float(loss_weight.cpu())
                except Exception:  # Handle division by zero/nan
                    loss_weight = torch.tensor(1.0).to(device)
                total_loss = loss_weight * node_loss + edge_loss

            forward_time = time.time() - start_time

            start_time = time.time()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            backward_time = time.time() - start_time

            # ----------- Gradient Norm Clipping -----------
            # Apply gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model_0.parameters(), gradient_clip_norm)

            # Update EMA model weights after the main model has been updated
            update_ema_weights(model_to_use, ema_model, ema_decay)

            if scheduler is not None:
                scheduler.step()

            if num_step > 0 and num_step % log_clear_step == 0:
                os.system('clear')

            current_lr = optimizer.param_groups[0]["lr"]
            if num_step % log_step == 0:
                if is_ddp_training:
                    prefix = f'[{rank+1}/{world_size}] '
                else:
                    prefix = ''

                print(f'{prefix}Epoch: {epoch},  Step: {num_step}, LR: {current_lr:.4e}, Loss: {total_loss:.4f} '
                      f'Node Loss: {node_loss:.4f}, Edge Loss: {edge_loss:.4f}')
                # print(f'[Timing] Data Load: {data_load_time:.4f}s | Forward: {forward_time:.4f}s | Backward: {backward_time:.4f}s')
                if writer is not None:
                    writer.add_scalar("train/lr", current_lr, num_step)
                    writer.add_scalar("train/total_loss", total_loss, num_step)
                    writer.add_scalar("train/node_loss", node_loss, num_step)
                    writer.add_scalar("train/edge_loss", edge_loss, num_step)

            if rank == 0:
                if num_step > 0 and num_step % eval_step == 0:
                    copy_ema_to_model(model_to_use, ema_model)
                    avg_val_loss, avg_val_node_loss, avg_val_edge_loss, node_f1, edge_f1 = \
                        evaluate_model(cfg, model_to_use, val_dataloader, node_criterion, edge_criterion, device)

                    print(f'Validation: Step: {num_step} ({batch_idx}/{len(train_dataloader)}), '
                          f'Val Loss: {avg_val_loss:.4f}, Val Node Loss: {avg_val_node_loss:.4f}, Val Edge Loss: {avg_val_edge_loss:.4f}, '
                          f'Node F1: {node_f1:.4f}, Edge F1: {edge_f1:.4f}')

                    avg_f1 = (node_f1 + edge_f1) / 2
                    final_model_path = os.path.join(save_dir,
                                                    f"0.0000_{avg_f1:.4f}_step_{num_step}_lr_{current_lr:.2e}_"
                                                    f"node_{node_f1:.4f}_edge_{edge_f1:.4f}.pt")
                    save_model_and_optimizer(model_to_use, optimizer, final_model_path)
                    print(f"Final model saved to: {final_model_path}")
                    writer.add_scalar("val/loss", avg_val_loss, num_step)
                    writer.add_scalar("val/Avg-F1", avg_f1, num_step)
                    writer.add_scalar("val/Node-F1", node_f1, num_step)
                    writer.add_scalar("val/Edge-F1", edge_f1, num_step)

                    # Optional validation dataset
                    if val_dataloader2 is not None:
                        avg_val_loss, avg_val_node_loss, avg_val_edge_loss, node_f1, edge_f1 = \
                            evaluate_model(cfg, model_to_use, val_dataloader2, node_criterion, edge_criterion, device)

                        print(f'Validation: Step: {num_step} ({batch_idx}/{len(train_dataloader)}), '
                              f'Val Loss: {avg_val_loss:.4f}, Val Node Loss: {avg_val_node_loss:.4f}, Val Edge Loss: {avg_val_edge_loss:.4f}, '
                              f'Node F1: {node_f1:.4f}, Edge F1: {edge_f1:.4f}')

                        avg_f1 = (node_f1 + edge_f1) / 2
                        final_model_path = os.path.join(save_dir2,
                                                        f"0.0000_{avg_f1:.4f}_step_{num_step}_lr_{current_lr:.2e}_"
                                                        f"node_{node_f1:.4f}_edge_{edge_f1:.4f}.pt")
                        save_model_and_optimizer(model_to_use, None, final_model_path)


                if num_step > 0 and num_step % save_step == 0:
                    copy_ema_to_model(model_to_use, ema_model)
                    final_model_path = os.path.join(save_dir, "latest_save_point.pt")
                    save_model_and_optimizer(model_to_use, optimizer, final_model_path)

            # Synchronization after rank 0 operations
            if is_ddp_training:
                dist.barrier()

            # --------------------------------------------
            # Auto-Restart Phase Transition
            # --------------------------------------------
            best_file = None
            if auto_restart_lr is not None and auto_restart_step is not None:
                if steps_in_phase >= auto_restart_step[restart_index]:
                    print(f"[Auto-Restart] Phase {restart_index} finished after {steps_in_phase} steps.")

                    # Find best checkpoint for this LR
                    target_lr = float(auto_restart_lr[restart_index])
                    best_file = None
                    best_f1 = -1.0

                    for fname in os.listdir(save_dir):
                        if not fname.endswith(".pt"):
                            continue
                        if f"lr_{target_lr:.2e}" not in fname:
                            continue
                        try:
                            f1_val = float(fname.split("_")[1])
                            full_path = os.path.join(save_dir, fname)
                            # skip if this checkpoint was already used
                            if full_path in used_restart_files:
                                continue
                            if f1_val > best_f1:
                                best_f1 = f1_val
                                best_file = full_path
                        except:
                            continue

                    if best_file is not None:
                        print(f"[Auto-Restart] Loading best checkpoint {best_file} (F1={best_f1:.4f})")
                        load_model_and_optimizer(model_to_use, best_file, optimizer=None)
                        ema_model = copy.deepcopy(model_to_use)
                        ema_model.eval()
                        used_restart_files.add(best_file)

                    # Move to next phase
                    restart_index = (restart_index + 1) % len(auto_restart_lr)
                    steps_in_phase = 0
                    target_lr = float(auto_restart_lr[restart_index])
                    set_lr(optimizer, target_lr)
                    print(
                        f"[Auto-Restart] Switched to LR={auto_restart_lr[restart_index]} for {auto_restart_step[restart_index]} steps")

                if is_ddp_training:
                    # Broadcast the new phase information (especially the loaded weights)
                    dist.barrier()

                    # Need to explicitly load the state dict on non-rank 0 if a checkpoint was loaded by rank 0
                    if rank != 0 and best_file is not None:
                        # Since all ranks hit the barrier, load the model now
                        load_model_and_optimizer(model_to_use, best_file, optimizer=None)
                        ema_model = copy.deepcopy(model_to_use)
                        ema_model.eval()

                if sleep_time > 0:
                    time.sleep(sleep_time)

    # --------------------------------------------
    # 6. Cleanup
    # --------------------------------------------
    if is_ddp_training:
        if rank == 0 and writer:
            writer.close()
        dist.destroy_process_group()


def calculate_class_balance():
    train_lmdb_path = cfg['train']['train_dataset']['lmdb_path']
    train_data = DocumentLMDBDataset(lmdb_path=train_lmdb_path, cache_size=0)

    node_count = [0] * len(cfg['data']['class_list'])
    edge_count = [0] * 2
    for idx, data in enumerate(train_data):
        if idx % 1000 == 0:
            print(f'{idx}/{len(train_data)}')

        for i in range(len(data.y)):
            node_count[data.y[i]] += 1
        for i in range(len(data.edge_label)):
            edge_count[data.edge_label[i]] += 1

    print('Node Count:')
    print(node_count)
    print('Edge Count:')
    print(edge_count)

    def calculate_balance(class_count):    # 방법 1: 단순 역수
        class_count = torch.tensor(class_count)
        weights_method1 = 1.0 / class_count
        # 가중치의 스케일을 조정하여 가장 작은 값이 1이 되도록 정규화 (선택 사항)
        weights_method1 = weights_method1 / torch.min(weights_method1)
        print("Weight (Method 1 - Normalized):", weights_method1.tolist())

        # 방법 2: 클래스별 빈도의 역수
        total_samples = torch.sum(class_count)
        weights_method2 = total_samples / (len(class_count) * class_count)
        print("Weight (Method 2):", weights_method2.tolist())

    print('Node Weight:')
    calculate_balance(node_count)

    print('Edge Weight:')
    calculate_balance(edge_count)


def safe_train(cfg, max_retries=1000, wait_sec=60):
    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Attempt {attempt} to run training...")
        proc = mp.Process(target=train, args=(cfg, 0, 1))
        proc.start()
        proc.join()

        if proc.exitcode == 0:
            print("[INFO] Training finished successfully.")
            break
        else:
            print(f"[WARN] Training crashed with exit code {proc.exitcode}. Retrying after {wait_sec} seconds...")
            time.sleep(wait_sec)
    else:
        print("[ERROR] Training failed after maximum retries.")


def is_ddp_run():
    """Checks if the script is being run in a Distributed Data Parallel environment."""
    # The essential DDP environment variables set by torchrun/torch.distributed.launch
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    # torch.autograd.set_detect_anomaly(True)

    if len(sys.argv) < 2:
        print("Usage: python train_gnn.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "rb") as f:
        cfg = anyconfig.load(f)

    if is_ddp_run():
        print("Distributed Data Parallel environment detected. Launching DDP training.")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK={rank}, WORLD_SIZE={world_size}')
    else:
        rank = 0
        world_size = 1

    # Original logic for single-process tasks
    task_name = cfg['train_task_name']
    if task_name == 'train':
        train(cfg, rank, world_size)
    elif task_name == 'safe_train':
        safe_train(cfg)
    elif task_name == 'calculate_class_balance':
        calculate_class_balance()
    else:
        print(f"Error: Unknown task_name '{task_name}' in config.")
        sys.exit(1)
