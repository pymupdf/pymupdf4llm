
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

from torch_geometric.loader import DataLoader as GeometricDataLoader
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


def evaluate_model(model, dataloader, node_criterion, edge_criterion, device):
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
    with open('train/cfgs/config.yaml', "rb") as f:
        cfg = anyconfig.load(f)

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

def train(cfg):
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

    train_dataloader = GeometricDataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           persistent_workers=False, drop_last=True)

    if val_dataset_type == 'json':
        val_json_path = cfg['train']['val_dataset']['json_path']
        val_data = DocumentJsonDataset(data_path=val_json_path, k=sample_k, class_map=data_class_map, is_train=True)
    elif train_dataset_type == 'lmdb':
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

    model = get_model(cfg, data_class_names)
    model = model.to(device)

    print_network(model, verbose=True)

    opt_type = cfg['train']['optimizer']['type']
    lr = float(cfg['train']['optimizer']['learning_rate'])

    if opt_type == 'SGD':
        momentum = float(cfg['train']['optimizer']['momentum'])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif opt_type == 'Adam':
        betas = cfg['train']['optimizer']['betas']
        weight_decay = float(cfg['train']['optimizer']['weight_decay'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=weight_decay, betas=betas)
    else:
        raise Exception(f'Invalid optimizer type = {opt_type}!')

    if load_from is not None and os.path.exists(load_from):
        print(f"Loading model from {load_from}...")
        opt = None
        if cfg['train']['load_optimizer']:
            opt = optimizer

        # Load the state dictionary with strict=False
        load_model_and_optimizer(model, load_from, optimizer=opt)
        print("Model loaded successfully!")

    ema_model = copy.deepcopy(model)
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

    # TensorBoard writer
    log_dir = os.path.join(save_dir, "tf_logs")
    writer = SummaryWriter(log_dir=log_dir)

    model.train()
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
            node_logits, edge_logits = model.forward_with_batch(batch)

            if node_loss_type == 'focal':
                soft_max = torch.nn.Softmax(dim=-1)
                node_logits = soft_max(node_logits)
            if edge_loss_type == 'focal':
                soft_max = torch.nn.Softmax(dim=-1)
                edge_logits = soft_max(edge_logits)

            node_loss = node_criterion(node_logits, batch.y)
            edge_loss = edge_criterion(edge_logits, batch.edge_label)

            if train_loss_weight is not None:
                total_loss = (train_loss_weight[0] * node_loss) + (train_loss_weight[1] * edge_loss)
            else:
                loss_weight = float((edge_loss / node_loss).cpu())
                total_loss = loss_weight * node_loss + edge_loss

            forward_time = time.time() - start_time

            # ignore nan-loss
            if torch.isnan(total_loss).any():
                continue

            start_time = time.time()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            backward_time = time.time() - start_time

            # ----------- Gradient Norm Clipping -----------
            # Apply gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            # Update EMA model weights after the main model has been updated
            update_ema_weights(model, ema_model, ema_decay)

            if scheduler is not None:
                scheduler.step()

            if num_step > 0 and num_step % log_clear_step == 0:
                os.system('clear')

            current_lr = optimizer.param_groups[0]["lr"]
            if num_step % log_step == 0:
                print(f'Epoch: {epoch}, Step: {num_step}, LR: {current_lr:.4e}, '
                      f'Loss: {total_loss:.4f}, Node Loss: {node_loss:.4f}, Edge Loss: {edge_loss:.4f}')
                # print(f'[Timing] Data Load: {data_load_time:.4f}s | Forward: {forward_time:.4f}s | Backward: {backward_time:.4f}s')
                writer.add_scalar("train/lr", current_lr, num_step)

            if num_step > 0 and num_step % eval_step == 0:
                copy_ema_to_model(model, ema_model)
                avg_val_loss, avg_val_node_loss, avg_val_edge_loss, node_f1, edge_f1 = \
                    evaluate_model(model, val_dataloader, node_criterion, edge_criterion, device)

                print(f'Validation: Step: {num_step} ({batch_idx}/{len(train_dataloader)}), '
                      f'Val Loss: {avg_val_loss:.4f}, Val Node Loss: {avg_val_node_loss:.4f}, Val Edge Loss: {avg_val_edge_loss:.4f}, '
                      f'Node F1: {node_f1:.4f}, Edge F1: {edge_f1:.4f}')

                avg_f1 = (node_f1 + edge_f1) / 2
                final_model_path = os.path.join(save_dir,
                                                f"0.0000_{avg_f1:.4f}_step_{num_step}_lr_{current_lr:.2e}_"
                                                f"node_{node_f1:.4f}_edge_{edge_f1:.4f}.pt")
                save_model_and_optimizer(model, optimizer, final_model_path)
                print(f"Final model saved to: {final_model_path}")
                writer.add_scalar("val/loss", avg_val_loss, num_step)
                writer.add_scalar("val/Avg-F1", avg_f1, num_step)
                writer.add_scalar("val/Node-F1", node_f1, num_step)
                writer.add_scalar("val/Edge-F1", edge_f1, num_step)

                # Optional validation dataset
                if val_dataloader2 is not None:
                    avg_val_loss, avg_val_node_loss, avg_val_edge_loss, node_f1, edge_f1 = \
                        evaluate_model(model, val_dataloader2, node_criterion, edge_criterion, device)

                    print(f'Validation: Step: {num_step} ({batch_idx}/{len(train_dataloader)}), '
                          f'Val Loss: {avg_val_loss:.4f}, Val Node Loss: {avg_val_node_loss:.4f}, Val Edge Loss: {avg_val_edge_loss:.4f}, '
                          f'Node F1: {node_f1:.4f}, Edge F1: {edge_f1:.4f}')

                    avg_f1 = (node_f1 + edge_f1) / 2
                    final_model_path = os.path.join(save_dir2,
                                                    f"0.0000_{avg_f1:.4f}_step_{num_step}_lr_{current_lr:.2e}_"
                                                    f"node_{node_f1:.4f}_edge_{edge_f1:.4f}.pt")
                    save_model_and_optimizer(model, None, final_model_path)

                model.train()

            if num_step > 0 and num_step % save_step == 0:
                copy_ema_to_model(model, ema_model)
                final_model_path = os.path.join(save_dir, "latest_save_point.pt")
                save_model_and_optimizer(model, optimizer, final_model_path)

            # --------------------------------------------
            # Auto-Restart Phase Transition
            # --------------------------------------------
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
                        # Only include files starting with "0.0000"
                        if not fname.startswith("0.0000"):
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
                        load_model_and_optimizer(model, best_file, optimizer=None)
                        ema_model = copy.deepcopy(model)
                        ema_model.eval()
                        used_restart_files.add(best_file)

                    # Move to next phase
                    restart_index = (restart_index + 1) % len(auto_restart_lr)
                    steps_in_phase = 0
                    target_lr = float(auto_restart_lr[restart_index])
                    # for g in optimizer.param_groups:
                    #     g['lr'] = float(target_lr)
                    print(
                        f"[Auto-Restart] Switched to LR={auto_restart_lr[restart_index]} for {auto_restart_step[restart_index]} steps")

                if sleep_time > 0:
                    time.sleep(sleep_time)

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


def safe_train(cfg, max_retries=1000, wait_sec=5):
    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Attempt {attempt} to run training...")
        proc = mp.Process(target=train, args=cfg)
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



def robins_features_extraction(feature_path, pdf_dir, filename, rect_list):
    import tempfile
    import subprocess

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write("PDF,page,x0,y0,x1,y1,class,score,order\n")
            n = 0
            for r in rect_list:
                line = f'{filename},0,{r[0]},{r[1]},{r[2]},{r[3]},0,0,{n}'
                n = n + 1
                tmp.write(line + '\n')
        command = '%s -d "%s" "%s"' % (feature_path, pdf_dir, path)
        result = subprocess.run(command, text=True, capture_output=True, shell=True)
        if result.returncode:
            print('Command failed:\n' + command + '\n')
        lines = result.stdout.splitlines()
        feature_rect_list = []
        feature_header = []

        for line_idx, line in enumerate(lines):
            if line_idx == 0:
                feature_header = line.split(',')
                continue
            features = []
            for x in line.split(','):
                features.append(x)
            feature_rect_list.append(features)
        return feature_header, feature_rect_list
    except Exception as ex:
        print('%s: %s' % (filename, ex))
    finally:
        os.remove(path)
    return None, None


def create_pdf_input_data(features_path, pdf_path):
    data_dict = {
        'bboxes': [],
        'text': [],
        'custom_features': []
    }
    box_type = []

    doc = pymupdf.open(pdf_path)
    page = doc[0]

    rects = [itm["bbox"] for itm in page.get_image_info()]
    for rect in rects:
        x1 = max(0, rect[0])
        y1 = max(0, rect[1])
        x2 = max(0, rect[2])
        y2 = max(0, rect[3])
        data_dict['bboxes'].append([x1, y1, x2, y2])
        data_dict['text'].append('')
        box_type.append('image')

    paths = [
        p for p in page.get_drawings() if p["rect"].width > 3 and p["rect"].height > 3
    ]
    vector_rects = page.cluster_drawings(drawings=paths)
    for vrect in vector_rects:
        x1 = vrect[0]
        y1 = vrect[1]
        x2 = vrect[2]
        y2 = vrect[3]
        data_dict['bboxes'].append([x1, y1, x2, y2])
        data_dict['text'].append('')
        box_type.append('vector')

    blocks = page.get_text("dict", flags=11)["blocks"]
    for block in blocks:
        for line in block["lines"]:
            x1 = line['bbox'][0]
            y1 = line['bbox'][1]
            x2 = line['bbox'][2]
            y2 = line['bbox'][3]

            txt = []
            for span in line['spans']:
                txt.append(span['text'])
            txt = ' '.join(txt).strip()
            data_dict['bboxes'].append([x1, y1, x2, y2])
            data_dict['text'].append(txt)
            box_type.append('line')

    pdf_dir = os.path.dirname(pdf_path)
    file_name = os.path.basename(pdf_path)
    rb_feat_name, rb_feat_value = robins_features_extraction(features_path, pdf_dir, file_name, data_dict['bboxes'])

    for row_idx in range(len(rb_feat_value)):
        custom_feature = {}
        for f_idx, f_name in enumerate(rb_feat_name):
            if f_idx < 9:
                continue
            custom_feature[f_name] = rb_feat_value[row_idx][f_idx]
        custom_feature['is_vector'] = box_type[row_idx] == 'vector'
        custom_feature['is_line'] = box_type[row_idx] == 'text'
        custom_feature['is_image'] = box_type[row_idx] == 'image'
        data_dict['custom_features'].append(custom_feature)

    return data_dict


def make_train_data_pymupdf(cfg):
    pdf_dir = cfg['make_train_data_pymupdf']['pdf_dir']
    pkl_dir = cfg['make_train_data_pymupdf']['pkl_dir']
    save_dir = cfg['make_train_data_pymupdf']['save_dir']
    features_path = cfg['make_train_data_pymupdf']['features_path']

    os.makedirs(save_dir, exist_ok=True)

    for file_idx, file_name in enumerate(os.listdir(pkl_dir)):
        pkl_file = f'{pkl_dir}/{file_name}'
        with open(pkl_file, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)
        cv_img = cv2.imdecode(pkl_data['jpeg'], cv2.IMREAD_COLOR)

        pdf_path = f'{pdf_dir}/{file_name[:-4]}.pdf'
        pdf_data = create_pdf_input_data(features_path, pdf_path)

        doc = pymupdf.open(pdf_path)
        page = doc[0]

        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes.reshape(pix.height, pix.width, pix.n)
        page_img = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        gt_label = pkl_data['label']
        for ant_idx, ant in enumerate(gt_label):
            x1 = ant[0] * (page_img.shape[1] / cv_img.shape[1])
            y1 = ant[1] * (page_img.shape[0] / cv_img.shape[0])
            x2 = ant[2] * (page_img.shape[1] / cv_img.shape[1])
            y2 = ant[3] * (page_img.shape[0] / cv_img.shape[0])

            cv2.putText(page_img, ant[-1], (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
            cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            ant[0] = x1
            ant[1] = y1
            ant[2] = x2
            ant[3] = y2

        bboxes = pdf_data['bboxes']
        for bbox_idx, bbox in enumerate(bboxes):
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            cv2.rectangle(page_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

        cv2.imshow('Page', page_img)
        cv2.waitKey()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    # torch.autograd.set_detect_anomaly(True)

    if len(sys.argv) < 2:
        print("Usage: python train_gnn.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "rb") as f:
        cfg = anyconfig.load(f)

    task_name = cfg['train_task_name']
    if task_name == 'train':
        train(cfg)
    elif task_name == 'safe_train':
        safe_train(cfg)
    elif task_name == 'make_train_data_pymupdf':
        make_train_data_pymupdf(cfg)
    elif task_name == 'calculate_class_balance':
        calculate_class_balance()
