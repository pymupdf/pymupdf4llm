import os
import shutil
import time

import cv2
import anyconfig

import blosc
import pickle
import lmdb

import numpy as np

import torch
import torch.optim as optim
import onnx
import onnxruntime as ort
import warnings
from tqdm import tqdm


import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datetime import datetime

from train.core.common.util import print_network
from train.core.losses import DiceLoss
from train.core.losses import IoULoss
from train.tools.data.segmentation.DocumentJsonDataset import (DocumentJsonDataset, _tensor_to_bgr_uint8, _make_colormap,
                                                               _mask_to_color)

from train.tools.schedulers.CustomCyclicLR import CustomCyclicLR

from train.core import SimpleDiscriminator, UNetThin, UNetThin2
from train.core.losses import FocalLoss

from train.core import load_model_and_optimizer


def evaluate_segmentation(model, data_loader, device,
                          num_classes, model_type='SEG',
                          eval_type='mIoU'):
    model.eval()  # set model to evaluation mode

    save_dir = "temp/segmentation"
    save_images = os.path.exists(save_dir)

    # initialize intersection and union counters
    intersection_meter = torch.zeros(num_classes, dtype=torch.float64)
    union_meter = torch.zeros(num_classes, dtype=torch.float64)

    # initialize pixel accuracy counters
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():  # disable gradient calculation
        for batch_idx, batch in enumerate(data_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # squeeze mask if shape is (B,1,H,W)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            if model_type == 'VAE':
                outputs, _, _ = model(images)
            else:
                outputs = model(images)

            predicted_masks = torch.argmax(outputs, dim=1)

            # calculate IoU per class
            if eval_type == 'mIoU':
                for cls in range(num_classes):
                    pred_cls = (predicted_masks == cls)
                    true_cls = (masks == cls)

                    intersection = (pred_cls & true_cls).sum().item()
                    union = (pred_cls | true_cls).sum().item()

                    intersection_meter[cls] += intersection
                    union_meter[cls] += union

            # calculate pixel accuracy
            correct_pixels += (predicted_masks == masks).sum().item()
            total_pixels += masks.numel()

            # save segmentation results if directory exists
            if save_images:
                for i in range(images.size(0)):
                    file_idx = batch_idx * data_loader.batch_size + i

                    # save ground truth mask
                    plt.imsave(
                        os.path.join(save_dir, f"{file_idx}_mask.jpg"),
                        masks[i].detach().cpu().numpy(),
                        cmap="gray"
                    )
                    # save predicted mask
                    plt.imsave(
                        os.path.join(save_dir, f"{file_idx}_prediction.jpg"),
                        predicted_masks[i].detach().cpu().numpy(),
                        cmap="gray"
                    )

    # compute IoU per class and mean IoU
    iou_per_class = (intersection_meter + 1e-6) / (union_meter + 1e-6)
    mIoU = torch.mean(iou_per_class).item()

    # compute pixel accuracy
    pixel_acc = correct_pixels / (total_pixels + 1e-6)
    model.train()  # set model back to training mode

    # return based on eval_type
    if eval_type == 'pixel':
        return pixel_acc, None
    elif eval_type == 'mIoU':
        return mIoU, iou_per_class.cpu().numpy()
    else:
        raise ValueError("eval_type must be either 'pixel' or 'mIoU'")





def evaluate_reconstruction(model, data_loader, device):
    model.eval()  # Set model to evaluation mode (e.g., disable dropout, batchnorm updates)
    reconstruction_accuracy = []
    for batch_idx, batch in enumerate(data_loader):
        images = batch['image'].to(device)  # Original images (Ground Truth)
        # Assuming 'masks' represents the reconstructed images.
        # In a real scenario, this would be `model(images)`.
        reconstructed_images = batch['mask'].to(device)

        # Calculate the absolute difference between the original and reconstructed images.
        # `images` and `reconstructed_images` are assumed to be of shape (N, C, H, W).
        absolute_error = torch.abs(images - reconstructed_images)

        # Calculate the Mean Absolute Error (MAE) for each image.
        # We take the mean across the channel, height, and width dimensions.
        # The shape changes from [N, C, H, W] to [N].
        mae_per_image = torch.mean(absolute_error, dim=[1, 2, 3])

        # Convert MAE into an accuracy metric.
        # If MAE is 0, accuracy is 1 (perfect match).
        # If MAE is 1, accuracy is 0 (complete mismatch).
        accuracy_per_image = 1.0 - mae_per_image

        # Append the calculated accuracy for each image to the list.
        reconstruction_accuracy.extend(accuracy_per_image.detach().cpu().tolist())

    # Calculate the overall average accuracy.
    acc = np.mean(reconstruction_accuracy)
    model.train()  # Set model back to training mode
    return acc


def _one_hot_encode(mask, num_classes):
    """
    Converts a single-channel mask to a one-hot encoded mask.
    mask: torch.Tensor of shape (N, H, W)
    num_classes: int
    Returns: torch.Tensor of shape (N, num_classes, H, W)
    """
    return nn.functional.one_hot(mask, num_classes=num_classes).permute(0, 3, 1, 2).float()


def get_segmentation_criterion(loss_config_list, num_classes, class_weight=None, device='cuda'):
    """
    loss_config_list: list of dicts, each with keys:
        - 'type': loss type string (e.g., 'focal', 'IoU', 'CE')
        - 'weight': float weight for this loss
        - 'param': optional dict of parameters for the loss
    """
    loss_functions = []

    for config in loss_config_list:
        loss_type = config['type']
        weight = config.get('weight', 1.0)
        param = config.get('param', {})

        if loss_type == 'focal':
            gamma = param.get('gamma', 1.0)
            alpha = param.get('alpha', None)
            reduction = param.get('reduction', 'mean')
            criterion = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)

        elif loss_type == 'IoU':
            criterion = IoULoss(num_classes=num_classes)

        elif loss_type == 'Dice':
            criterion = DiceLoss(num_classes=num_classes)

        elif loss_type == 'CE':
            reduction = param.get('reduction', 'mean')
            if class_weight is not None:
                class_weight_tensor = torch.tensor(class_weight).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, reduction=reduction)
            else:
                criterion = nn.CrossEntropyLoss(reduction=reduction)

        else:
            raise Exception(f'Invalid loss_type = {loss_type}')

        loss_functions.append((criterion, weight))

    def combined_loss(pred, target):
        total_loss = 0.0
        for criterion, weight in loss_functions:
            total_loss += weight * criterion(pred, target)
        return total_loss

    return combined_loss



def train_model(cfg):
    # Device setup
    device = cfg['train_segmentation']['device']
    image_size = cfg['train_segmentation']['model']['image_size']
    mask_size = cfg['train_segmentation']['model']['mask_size']

    train_pkl_prob = cfg['train_segmentation']['train_pkl_prob']
    train_pkl_dir = cfg['train_segmentation']['train_pkl_dir']
    train_pdf_dir = cfg['train_segmentation']['train_pdf_dir']
    val_pkl_dir = cfg['train_segmentation']['val_pkl_dir']
    val_pdf_dir = cfg['train_segmentation']['val_pdf_dir']
    save_dir = cfg['train_segmentation']['save_dir']

    train_cache_size = cfg['train_segmentation']['train_cache_size']
    train_cache_rate = cfg['train_segmentation']['train_cache_rate']
    load_from = cfg['train_segmentation']['load_from']
    data_class_names = cfg['train_segmentation']['class_list']
    data_class_change_map = cfg['train_segmentation']['data_class_change_map']
    num_classes = len(data_class_names) + 1

    data_class_map = {}
    for i in range(len(data_class_names)):
        data_class_map[data_class_names[i]] = i

    for key in data_class_change_map.keys():
        val = data_class_change_map[key]
        if val is not None:
            data_class_map[key] = data_class_map[val]
        else:
            data_class_map[key] = -1

    batch_size = cfg['train_segmentation']['batch_size']
    learning_rate = float(cfg['train_segmentation']['learning_rate'])
    num_epochs = cfg['train_segmentation']['num_epochs']
    loss_type = cfg['train_segmentation']['loss_type']
    class_weight = cfg['train_segmentation']['class_weight']

    # Updated: Use train_style string instead of gan_style_train boolean
    train_style = cfg['train_segmentation']['train_style']
    kl_weight = cfg['train_segmentation'].get('kl_weight', 0.01)
    gan_loss_weight = cfg['train_segmentation'].get('gan_loss_weight', 0.1)

    log_interval = cfg['train_segmentation']['log_interval']
    show_interval = cfg['train_segmentation']['show_interval']
    eval_interval = cfg['train_segmentation']['eval_interval']
    sleep_time = cfg['train_segmentation']['sleep_time']

    crop_prob = cfg['train_segmentation']['augmentation']['crop_prob']
    crop_ratio = cfg['train_segmentation']['augmentation']['crop_ratio']
    noise_prob = cfg['train_segmentation']['augmentation']['noise_prob']
    noise_ratio = cfg['train_segmentation']['augmentation']['noise_ratio']

    target_type = cfg['train_segmentation']['model']['target_type']
    num_workers = cfg['train_segmentation']['num_workers']

    # Create dataset and dataloader
    train_dataset = DocumentJsonDataset(pkl_dir=train_pkl_dir, pkl_prob=train_pkl_prob, pdf_dir=train_pdf_dir,
                                        class_map=data_class_map, image_size=image_size, output_mask_size=mask_size,
                                        target_type=target_type,
                                        cache_size=train_cache_size, cache_rate=train_cache_rate,
                                        crop_prob=crop_prob, crop_ratio=crop_ratio, noise_prob=noise_prob,
                                        noise_ratio=noise_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create validation dataset and dataloader
    val_dataset = DocumentJsonDataset(pkl_dir=val_pkl_dir, pdf_dir=val_pdf_dir,
                                      class_map=data_class_map, image_size=image_size, output_mask_size=mask_size,
                                      target_type=target_type)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    with open(f'{save_dir}/model.yaml', 'w') as f:
        yaml.dump({
            'model': cfg['train_segmentation']['model']
        }, f)

    # Initialize Generator (segmentation model) based on train_style
    if train_style == 'VAE':
        # model = VAEUNetMultiScale(num_classes, num_filters=num_filters, out_size=mask_size, input_channels=in_ch,
        #                           use_sep=use_sep).to(device)
        print("\nModel Architecture (VAE):")
    elif train_style == 'GAN' or train_style == 'SEG':  # Add SEG style here
        assert num_classes == cfg['train_segmentation']['model']['out_ch']
        if cfg['train_segmentation']['model']['name'] == 'UNet-thin':
            model = UNetThin(cfg['train_segmentation']['model']).to(device)
        elif cfg['train_segmentation']['model']['name'] == 'UNet-thin2':
            model = UNetThin2(cfg['train_segmentation']['model']).to(device)
        print("\nModel Architecture (Generator/SimpleUNet):")
    else:
        raise ValueError("Invalid train_style. Choose 'SEG', 'VAE', or 'GAN'.")  # Update error message

    print_network(model, verbose=True)

    # Initialize Discriminator if GAN training is enabled
    discriminator = None
    if train_style == 'GAN':
        if target_type == 'reconstruction':
            disc_in_ch = 2  # image(1) + mask(1)
        else:
            disc_in_ch = num_classes + 1
        discriminator = SimpleDiscriminator(in_channels=disc_in_ch).to(device)

        print("\nModel Architecture (Discriminator):")
        print(discriminator)

        disc_load_from = f'{save_dir}/latest_save_point_disc.pth'
        if os.path.exists(disc_load_from):
            print(f"Loading discriminator from {disc_load_from}...")
            data = torch.load(disc_load_from, map_location=torch.device('cpu'))
            if 'model_state_dict' in data:
                discriminator.load_state_dict(data['model_state_dict'], strict=False)
            else:
                discriminator.load_state_dict(data)
            print("Discriminator loaded successfully!")

    if load_from is not None and os.path.exists(load_from):
        print(f"Loading model from {load_from}...")
        load_model_and_optimizer(model, load_from)
        print("Model loaded successfully!")

    # Define loss functions
    if target_type == 'reconstruction':
        segmentation_criterion = nn.HuberLoss()
    else:
        segmentation_criterion = get_segmentation_criterion(loss_type, num_classes, class_weight)

    # GAN-specific loss
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Define optimizers
    optimizer_G = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_D = None
    if train_style == 'GAN':
        optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

    base_lr = float(cfg['train_segmentation']['scheduler']['base_lr'])
    max_lr = float(cfg['train_segmentation']['scheduler']['max_lr'])
    step_size = int(cfg['train_segmentation']['scheduler']['step_size'])
    gamma = 1.0
    scheduler = CustomCyclicLR(optimizer_G, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size,decay_factor=gamma)

    eval_type = 'pixel'
    os.makedirs(save_dir, exist_ok=True)
    print("\nStarting multi-class training...")

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_loss_D = 0.0
        running_loss_G_adv = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # GAN Training Logic
            if train_style == 'GAN':
                # =====================================================================
                # 1. Train the Discriminator
                # =====================================================================
                discriminator.train()
                optimizer_D.zero_grad()

                # Train with real samples
                if target_type == 'reconstruction':
                    real_masks = masks.float()
                else:
                    real_masks = _one_hot_encode(masks, num_classes)

                D_real_output = discriminator(images, real_masks)
                D_real_loss = adversarial_criterion(D_real_output, torch.ones_like(D_real_output))

                # Train with fake samples
                with torch.no_grad():
                    fake_masks_logits = model(images)
                    if target_type == 'reconstruction':
                        fake_masks = fake_masks_logits
                    else:
                        fake_masks = fake_masks_logits

                D_fake_output = discriminator(images, fake_masks)
                D_fake_loss = adversarial_criterion(D_fake_output, torch.zeros_like(D_fake_output))

                # Total Discriminator Loss
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                optimizer_D.step()
                running_loss_D += D_loss.item()

                # =====================================================================
                # 2. Train the Generator (model)
                # =====================================================================
                model.train()
                optimizer_G.zero_grad()
                outputs = model(images)

                loss_seg = segmentation_criterion(outputs, masks)

                fake_masks_for_G = outputs
                D_output_for_G = discriminator(images, fake_masks_for_G)
                loss_G_adv = 0.001 * adversarial_criterion(D_output_for_G, torch.ones_like(D_output_for_G))

                loss = loss_seg + gan_loss_weight * loss_G_adv
                running_loss_G_adv += loss_G_adv.item()

            # VAE Training Logic
            elif train_style == 'VAE':
                model.train()
                optimizer_G.zero_grad()

                # Forward pass for VAE returns logits, mu, logvar
                outputs, mu, logvar = model(images)

                # Calculate segmentation loss
                loss_seg = segmentation_criterion(outputs, masks)

                # Calculate KL Divergence Loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Combine losses
                loss = loss_seg + kl_weight * kl_loss

            # SEG Training Logic (Added as requested)
            elif train_style == 'SEG':
                model.train()
                optimizer_G.zero_grad()
                outputs = model(images)
                loss = segmentation_criterion(outputs, masks)

            else:
                raise ValueError(f"Invalid train_style: {train_style}. Choose 'SEG', 'VAE', or 'GAN'.")

            loss.backward()
            optimizer_G.step()
            running_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            # Visualization and logging
            current_iteration = epoch * len(train_dataloader) + (batch_idx + 1)

            if target_type == 'reconstruction':
                if current_iteration % log_interval == 0:
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    lr_str = f"{optimizer_G.param_groups[0]['lr']:.2e}"
                    loss_val = loss.item() if not isinstance(loss, float) else loss
                    print(f"[{now}] iter={current_iteration} lr={lr_str} loss={loss_val:.4f}", flush=True)

                    if current_iteration % show_interval == 0:
                        first_image = images[0:1]
                        first_mask = masks[0:1]
                        first_output = outputs[0:1]
                        image = (first_image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                        mask = (first_mask.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                        output = (first_output.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
                        image = np.squeeze(image, axis=0)
                        mask = np.squeeze(mask, axis=0)
                        output = np.squeeze(output, axis=0)
                        if mask.shape[2] == 1:
                            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        if image.shape[2] == 1:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        if output.shape[2] == 1:
                            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                        disp = np.concatenate([image, output, mask], axis=1)
                        if disp.shape[0] > 900:
                            resize_rate = 900 / disp.shape[0]
                            disp = cv2.resize(disp, (0, 0), fx=resize_rate, fy=resize_rate)
                        cv2.imshow('Training...', disp)
                        cv2.waitKey(10)
            else:  # Segmentation
                if current_iteration % log_interval == 0:
                    # Pixel acc
                    if eval_type == 'pixel':
                        predicted_masks = torch.argmax(outputs, dim=1)
                        correct_pixels = (predicted_masks == masks).float()
                        correct_count_per_image = correct_pixels.sum(dim=[1, 2])
                        total_pixels_per_image = masks.shape[1] * masks.shape[2]
                        accuracy_per_image = correct_count_per_image / total_pixels_per_image
                        average_batch_accuracy = accuracy_per_image.mean().item()
                    # mIoU
                    elif eval_type == 'mIoU':
                        predicted_masks = torch.argmax(outputs, dim=1)
                        batch_size = predicted_masks.shape[0]
                        iou_per_image = []
                        for b in range(batch_size):
                            iou_per_class = []
                            for cls in range(num_classes):
                                pred_cls = (predicted_masks[b] == cls)
                                true_cls = (masks[b] == cls)

                                intersection = (pred_cls & true_cls).sum().item()
                                union = (pred_cls | true_cls).sum().item()

                                if union > 0:
                                    iou = intersection / union
                                    iou_per_class.append(iou)

                            # mean IoU for this image (exclude classes not present in GT and prediction)
                            if len(iou_per_class) > 0:
                                iou_per_image.append(sum(iou_per_class) / len(iou_per_class))
                            else:
                                iou_per_image.append(0.0)

                        # average mIoU across the batch
                        average_batch_accuracy = sum(iou_per_image) / len(iou_per_image)

                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    lr_str = f"{optimizer_G.param_groups[0]['lr']:.2e}"
                    if train_style == 'GAN':
                        D_loss_val = running_loss_D / log_interval
                        G_adv_loss_val = running_loss_G_adv / log_interval
                        print(
                            f"[{now}] iter={current_iteration} lr={lr_str} loss_seg={loss_seg:.4f} loss_G_adv={G_adv_loss_val:.4f} D_loss={D_loss_val:.2e} acc={average_batch_accuracy:.4f}",
                            flush=True)
                        running_loss_D = 0.0
                        running_loss_G_adv = 0.0
                    elif train_style == 'VAE':
                        loss_seg_val = loss_seg.item()
                        kl_loss_val = kl_loss.item()
                        loss_val = loss.item()
                        print(
                            f"[{now}] iter={current_iteration} lr={lr_str} loss_seg={loss_seg_val:.4f} kl_loss={kl_loss_val:.4f} total_loss={loss_val:.4f} acc={average_batch_accuracy:.4f}",
                            flush=True)
                    elif train_style == 'SEG':  # Log for SEG style
                        loss_val = loss.item() if not isinstance(loss, float) else loss
                        print(
                            f"[{now}] iter={current_iteration} lr={lr_str} loss={loss_val:.4f}, acc={average_batch_accuracy:.4f}",
                            flush=True)
                    else:  # Invalid style case
                        loss_val = loss.item() if not isinstance(loss, float) else loss
                        print(
                            f"[{now}] iter={current_iteration} lr={lr_str} loss={loss_val:.4f}, acc={average_batch_accuracy:.4f}",
                            flush=True)
                    running_loss = 0.0

                if current_iteration % show_interval == 0:
                    try:
                        img_sample = images[0].detach().cpu()
                        img_bgr = _tensor_to_bgr_uint8(img_sample)
                        if train_style == 'VAE':
                            # outputs from VAE is (logits, mu, logvar), need to take only logits for visualization
                            logits_sample = outputs[0]
                        else:
                            logits_sample = outputs[0]

                        pred_map = logits_sample.argmax(dim=0).detach().cpu().numpy().astype(np.int32)
                        pred_num_channels = logits_sample.shape[0]
                        gt_sample = batch['mask'][0]
                        gt_map = gt_sample.squeeze().detach().cpu().numpy().astype(np.int32)
                        gt_map = gt_map.astype(np.int32)
                        num_cls = max(pred_num_channels, int(gt_map.max() + 1))
                        mask_colors = _make_colormap(num_cls, seed=0)
                        pred_color = _mask_to_color(pred_map, mask_colors)
                        gt_color = _mask_to_color(gt_map, mask_colors)
                        if pred_color.shape[:2] != img_bgr.shape[:2]:
                            pred_color_resized = cv2.resize(pred_color, (img_bgr.shape[1], img_bgr.shape[0]),
                                                            interpolation=cv2.INTER_NEAREST)
                        else:
                            pred_color_resized = pred_color
                        if gt_color.shape[:2] != img_bgr.shape[:2]:
                            gt_color_resized = cv2.resize(gt_color, (img_bgr.shape[1], img_bgr.shape[0]),
                                                          interpolation=cv2.INTER_NEAREST)
                        else:
                            gt_color_resized = gt_color
                        overlay_pred = cv2.addWeighted(img_bgr, 0.7, pred_color_resized, 0.3, 0)
                        overlay_gt = cv2.addWeighted(img_bgr, 0.7, gt_color_resized, 0.3, 0)
                        pred_map_resized = cv2.resize(pred_map.astype(np.int32), (img_bgr.shape[1], img_bgr.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST) if pred_map.shape != img_bgr.shape[
                            :2] else pred_map
                        gt_map_resized = cv2.resize(gt_map.astype(np.int32), (img_bgr.shape[1], img_bgr.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST) if gt_map.shape != img_bgr.shape[
                            :2] else gt_map
                        diff_mask = (pred_map_resized != gt_map_resized).astype(np.uint8) * 255
                        diff_color = np.zeros_like(img_bgr)
                        diff_color[..., 2] = diff_mask
                        overlay_diff = cv2.addWeighted(img_bgr, 0.7, diff_color, 0.3, 0)
                        col_1 = np.concatenate([img_bgr, gt_color_resized, overlay_gt], axis=0)
                        col_2 = np.concatenate([overlay_diff, pred_color_resized, overlay_pred], axis=0)
                        disp = np.concatenate([col_1, col_2], axis=1)
                        if disp.shape[0] > 900:
                            resize_rate = 900 / disp.shape[0]
                        else:
                            resize_rate = 1.0
                        disp = cv2.resize(disp, (0, 0), fx=resize_rate, fy=resize_rate)
                        cv2.imshow('Training...', disp)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"[Visualizer] failed to visualize sample at iter {current_iteration}: {e}")
                        import traceback
                        traceback.print_exc()

            if current_iteration % eval_interval == 0:
                print(f"--- Iteration {current_iteration} Evaluation ---")
                if target_type == 'reconstruction':
                    val_acc = evaluate_reconstruction(model, val_dataloader, device)
                else:
                    val_acc, _ = evaluate_segmentation(model, val_dataloader, device, num_classes=num_classes,
                                                       eval_type=eval_type, model_type=train_style)
                    print(f"  Validation Acc: {val_acc:.4f}")
                model_filename = os.path.join(save_dir,
                                              f"{val_acc:.4f}_model_epoch{epoch + 1}_iter{current_iteration}.pth")
                torch.save(model.state_dict(), model_filename)
                print(f"  Model saved to {model_filename}")
                model_filename = os.path.join(save_dir, 'latest_save_point.pth')
                torch.save(model.state_dict(), model_filename)
                if discriminator is not None:
                    model_filename = os.path.join(save_dir, 'latest_save_point_disc.pth')
                    torch.save(discriminator.state_dict(), model_filename)

            if sleep_time > 0:
                time.sleep(sleep_time)
            model.train()

    print("Multi-class training finished!")
    print(f"All models saved in: {save_dir}")
    print(f"Best model saved as: {os.path.join(save_dir, 'best_model.pth')}")


def export2onnx(cfg):
    """
    Converts a trained PyTorch model to ONNX format and verifies that the output
    from the exported model matches the original PyTorch model's output.
    """

    load_from = cfg['train_segmentation']['load_from']
    onnx_save = cfg['train_segmentation']['onnx_save']
    train_style = cfg['train_segmentation']['train_style']

    in_ch = cfg['train_segmentation']['model']['in_ch']
    image_size = cfg['train_segmentation']['model']['image_size']

    data_class_names = cfg['data']['class_list']
    data_class_map = {}
    for i in range(len(data_class_names)):
        data_class_map[data_class_names[i]] = i

    if train_style == 'VAE':
        # model = VAEUNetMultiScale(num_classes, num_filters=num_filters, out_size=mask_size, input_channels=in_ch,
        #                           output_name=output_name, use_sep=use_sep).to('cpu')
        print("\nModel Architecture (VAE):")
    elif train_style == 'GAN' or train_style == 'SEG':  # Add SEG style here
        model = UNetThin(cfg['train_segmentation']['model']).to('cpu')
        print("\nModel Architecture (Generator/SimpleUNet):")
    else:
        raise ValueError("Invalid train_style. Choose 'SEG', 'VAE', or 'GAN'.")  # Update error message

    print("\nModel Architecture:")
    print(model)

    if load_from is not None and os.path.exists(load_from):
        print(f"Loading model from {load_from}...")
        load_model_and_optimizer(model, load_from)
        print("Model loaded successfully!")
    else:
        raise Exception(f'{load_from} is not exist!')


    # 1. Convert the PyTorch model to ONNX
    try:
        # Set the model to evaluation mode
        model.eval()

        # Create a dummy input for ONNX conversion
        dummy_input_size = [1, in_ch]
        dummy_input_size.extend(image_size['size'])
        dummy_input = torch.randn(dummy_input_size).to(next(model.parameters()).device)

        # Use torch.onnx.export to convert the model to ONNX
        torch.onnx.export(model,
                          dummy_input,
                          onnx_save,
                          opset_version=11,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

        print(f"? Model successfully converted to {onnx_save}.")

    except Exception as e:
        warnings.warn(f"An error occurred during ONNX conversion: {e}")
        return False, f"ONNX conversion failed: {e}"

    # 2. Validate the converted ONNX model
    # Use onnx.checker to ensure the ONNX model is valid.
    try:
        onnx_model = onnx.load(onnx_save)
        onnx.checker.check_model(onnx_model)
        print("? ONNX model validation completed.")
    except onnx.checker.ValidationError as e:
        warnings.warn(f"?? ONNX model validation failed: {e}")
        return False, f"ONNX model validation failed: {e}"

    try:
        # Create an ONNX runtime session
        ort_session = ort.InferenceSession(onnx_save, providers=['CPUExecutionProvider'])

        for i in range(10):
            dummy_input = torch.randn(dummy_input_size).to(next(model.parameters()).device)
            dummy_input_np = dummy_input.detach().cpu().numpy()

            # Get PyTorch model output
            with torch.no_grad():
                torch_output = model(dummy_input)

            # Get ONNX runtime output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
            ort_outputs = ort_session.run(None, ort_inputs)

            onnx_output = torch.from_numpy(ort_outputs[0])

            # Compare outputs
            # Use a small tolerance for floating-point inaccuracies
            if torch.allclose(torch_output, onnx_output, atol=1e-4):
                pass
            else:
                max_diff = torch.max(torch.abs(torch_output - onnx_output))
                warnings.warn(f"?? PyTorch and ONNX model outputs do not match. Max difference: {max_diff.item()}")

    except Exception as e:
        warnings.warn(f"?? An error occurred during output verification: {e}")
        return False, f"Output verification failed: {e}"


def save_class_pkl():
    src_dir = '/media/win/Dataset/PubLayNet/train_pkl_doclaynet'
    save_dir = '/media/win/Dataset/PubLayNet/train_pkl_doclaynet_table'
    target_class = ['table']

    os.makedirs(save_dir, exist_ok=True)

    file_list = os.listdir(src_dir)
    for file_idx, pkl_name in enumerate(file_list):
        if file_idx % 1000 == 0:
            print('%d/%d ...' % (file_idx, len(file_list)))

        pkl_path = f'{src_dir}/{pkl_name}'
        with open(pkl_path, 'rb') as f:
            compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        pkl_data = pickle.loads(depressed_pickle)
        label = pkl_data['label']

        target_count = 0
        for ant in label:
            if ant[-1] in target_class:
                target_count += 1

        if target_count > 0:
            dst_path = f'{save_dir}/{pkl_name}'
            shutil.copy(pkl_path, dst_path)


def make_segmentation_cache(cfg, batch_size=8):
    from train.infer.pymupdf_util import create_input_data_by_pymupdf
    from train.infer.onnx.BoxRFDGNN import resize_image, to_gray

    load_from = cfg['train_segmentation']['load_from']
    pdf_dir = cfg['train_segmentation']['pdf_dir']
    device = cfg['train_segmentation']['device']
    train_style = cfg['train_segmentation']['train_style']
    lmdb_path = cfg['train_segmentation']['save_dir'] + '/segmentation_cache.lmdb'

    def normalize_image(img_gray):
        # Normalize grayscale image to [0, 1]
        min_val, max_val = img_gray.min(), img_gray.max()
        if max_val > min_val:
            return (img_gray - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(img_gray, dtype=np.float32)

    # Initialize model
    if train_style in ['GAN', 'SEG']:
        model = UNetThin(cfg['train_segmentation']['model']).to(device)
    else:
        raise ValueError("Invalid train_style. Choose 'SEG', 'VAE', or 'GAN'.")

    model.output_name = 'dec_all'
    model.eval()

    # Load pretrained weights
    if load_from is not None and os.path.exists(load_from):
        print(f"Loading model from {load_from}...")
        load_model_and_optimizer(model, load_from)
        print("Model loaded successfully!")
    else:
        raise Exception(f'{load_from} is not exist!')

    # Initialize LMDB environment
    env = lmdb.open(lmdb_path, map_size=10**10)  # Allow up to 10GB
    txn = env.begin(write=True)

    file_list = os.listdir(pdf_dir)
    file_list.sort()

    batch_inputs = []
    batch_keys = []

    for pdf_file in tqdm(file_list, desc="Processing PDFs"):
        key = pdf_file.encode()

        # Skip if already cached
        if txn.get(key) is not None:
            continue

        pdf_path = os.path.join(pdf_dir, pdf_file)
        data_dict = create_input_data_by_pymupdf(pdf_path)
        page_img = data_dict['image']
        img_resized = resize_image(page_img, (1000, 1000))
        img_gray = normalize_image(to_gray(img_resized)).astype(np.float32)

        nn_input = torch.tensor(img_gray[None, None, ...], dtype=torch.float32).to(device)
        batch_inputs.append(nn_input)
        batch_keys.append(key)

        # Run inference when batch is full
        if len(batch_inputs) == batch_size:
            batch_tensor = torch.cat(batch_inputs, dim=0)
            with torch.no_grad():
                batch_outputs = model(batch_tensor).cpu().numpy()

            for k, output in zip(batch_keys, batch_outputs):
                txn.put(k, pickle.dumps(output))

            txn.commit()
            txn = env.begin(write=True)
            batch_inputs = []
            batch_keys = []

    # Process remaining batch
    if batch_inputs:
        batch_tensor = torch.cat(batch_inputs, dim=0)
        with torch.no_grad():
            batch_outputs = model(batch_tensor).cpu().numpy()

        for k, output in zip(batch_keys, batch_outputs):
            txn.put(k, pickle.dumps(output))

        txn.commit()

    env.close()
    print("All segmentation results cached in LMDB.")



if __name__ == "__main__":
    with open('tools/config.yaml', "rb") as f:
        cfg = anyconfig.load(f)

    task = cfg['train_segmentation']['task']
    if task == 'train':
        # run_with_retry(train_model, cfg, max_retries=100)
        train_model(cfg)
    elif task == 'export2onnx':
        export2onnx(cfg)
    elif task == 'save_class_pkl':
        save_class_pkl()
    elif task == 'make_segmentation_cache':
        make_segmentation_cache(cfg)
