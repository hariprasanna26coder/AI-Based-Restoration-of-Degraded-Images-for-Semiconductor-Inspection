"""
train.py
--------
Training script for SemiRestoreNet.

Usage:
    python train.py                              # uses default config
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --resume saved_models/checkpoint_epoch_50.pt

What this script does:
  1. Loads config YAML
  2. Builds model, optimizer, scheduler, loss
  3. Loads train/val DataLoaders with augmentation
  4. Trains with mixed precision (AMP)
  5. Validates each epoch → computes PSNR + SSIM
  6. Saves best model by SSIM; saves periodic checkpoints
  7. Early stopping if validation stagnates
  8. Logs to CSV and prints progress
"""

import os
import sys
import csv
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

# ── try importing yaml ─────────────────────────────────────────────────────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("[WARNING] PyYAML not found. Using default config dict.")

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from model.network import SemiRestoreNet, count_params
from model.losses  import RestorationLoss
from data.dataset  import get_dataloaders
from utils.metrics import compute_metrics, AverageMeter
from utils.visualization import save_comparison_grid


# ---------------------------------------------------------------------------
# Default config (used if YAML not found)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    'data': {
        'train_noisy_dir' : 'train/train/NoisyLR',
        'train_gt_dir'    : 'train/train/GT',
        'val_split'       : 0.10,
        'noisy_patch_size': 64,
        'clip_sigma'      : 3.0,
        'num_workers'     : 0,
    },
    'model': {
        'in_ch'             : 1,
        'out_ch'            : 1,
        'base_ch'           : 64,
        'bottleneck_blocks' : 6,
        'scale'             : 2,
    },
    'training': {
        'epochs'     : 200,
        'batch_size' : 8,
        'seed'       : 42,
    },
    'optimizer': {
        'type'         : 'AdamW',
        'lr'           : 2e-4,
        'weight_decay' : 1e-4,
        'betas'        : [0.9, 0.999],
    },
    'scheduler': {
        'type'   : 'CosineAnnealingWarmRestarts',
        'T_0'    : 50,
        'T_mult' : 1,
        'eta_min': 1e-6,
    },
    'loss': {
        'w_char'  : 1.0,
        'w_ssim'  : 0.2,
        'w_edge'  : 0.1,
        'char_eps': 1e-3,
    },
    'amp': {'enabled': True},
    'checkpoint': {
        'save_dir'    : 'weights',
        'save_every'  : 10,
        'keep_last_n' : 3,
        'best_metric' : 'ssim',
    },
    'early_stopping': {
        'enabled'  : True,
        'patience' : 30,
        'min_delta': 1e-4,
    },
    'logging': {
        'log_every'     : 10,
        'val_every'     : 1,
        'save_vis_every': 10,
        'vis_dir'       : 'outputs/train_vis',
        'log_file'      : 'outputs/train_log.csv',
    },
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    if not HAS_YAML or not os.path.exists(path):
        print(f"[Config] Using DEFAULT config (YAML not found at {path})")
        return DEFAULT_CONFIG
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"[Config] Loaded from {path}")
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_old_checkpoints(save_dir: str, keep_n: int):
    """Keep only the `keep_n` most recent epoch checkpoints."""
    ckpts = sorted([
        f for f in os.listdir(save_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
    ])
    while len(ckpts) > keep_n:
        old = os.path.join(save_dir, ckpts.pop(0))
        os.remove(old)
        print(f"[Checkpoint] Removed old checkpoint: {old}")


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, amp_enabled, log_every
):
    model.train()
    loss_meter = AverageMeter('loss')
    char_meter = AverageMeter('char')
    ssim_meter = AverageMeter('ssim_l')
    edge_meter = AverageMeter('edge')

    start = time.time()
    for batch_idx, batch in enumerate(loader):
        noisy = batch['noisy'].to(device, non_blocking=True)
        gt    = batch['gt'].to(device,    non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=amp_enabled):
            pred = model(noisy)
            total, (lc, ls, le) = criterion(pred, gt)

        if torch.isnan(total) or torch.isinf(total):
            continue

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = noisy.size(0)
        loss_meter.update(total.item(), bs)
        char_meter.update(lc.item(), bs)
        ssim_meter.update(ls.item(), bs)
        edge_meter.update(le.item(), bs)

        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - start
            print(
                f"  [{batch_idx+1:4d}/{len(loader)}]"
                f"  loss={loss_meter.avg:.4f}"
                f"  char={char_meter.avg:.4f}"
                f"  ssim_l={ssim_meter.avg:.4f}"
                f"  edge={edge_meter.avg:.4f}"
                f"  ({elapsed:.1f}s)"
            )

    return {
        'loss'  : loss_meter.avg,
        'char'  : char_meter.avg,
        'ssim_l': ssim_meter.avg,
        'edge'  : edge_meter.avg,
    }


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device, amp_enabled):
    model.eval()
    loss_meter = AverageMeter('val_loss')
    psnr_meter = AverageMeter('psnr')
    ssim_meter = AverageMeter('ssim')

    sample_noisy = sample_pred = sample_gt = None

    for batch in loader:
        noisy = batch['noisy'].to(device, non_blocking=True)
        gt    = batch['gt'].to(device,    non_blocking=True)

        with autocast(device_type='cuda', enabled=amp_enabled):
            pred = model(noisy)
            pred_clamped = torch.clamp(pred, 0.0, 1.0)
            total, _ = criterion(pred_clamped, gt)

        bs = noisy.size(0)
        loss_meter.update(total.item(), bs)

        metrics = compute_metrics(pred_clamped, gt)
        psnr_meter.update(metrics['psnr'], bs)
        ssim_meter.update(metrics['ssim'], bs)

        # Keep one batch for visualization
        if sample_noisy is None:
            sample_noisy = noisy.cpu()
            sample_pred  = pred_clamped.cpu()
            sample_gt    = gt.cpu()

    return {
        'val_loss': loss_meter.avg,
        'psnr'    : psnr_meter.avg,
        'ssim'    : ssim_meter.avg,
    }, (sample_noisy, sample_pred, sample_gt)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train SemiRestoreNet')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint .pt file to resume from')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Seed ──────────────────────────────────────────────────────────────
    seed = cfg['training'].get('seed', 42)
    set_seed(seed)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = cfg['amp'].get('enabled', True) and device.type == 'cuda'
    print(f"[Device] Using: {device}  |  AMP: {amp_enabled}")

    # ── Output dirs ───────────────────────────────────────────────────────
    save_dir = cfg['checkpoint']['save_dir']
    vis_dir  = cfg['logging']['vis_dir']
    log_file = cfg['logging']['log_file']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    dcfg = cfg['data']
    train_loader, val_loader, _, _ = get_dataloaders(
        noisy_dir       = dcfg['train_noisy_dir'],
        gt_dir          = dcfg['train_gt_dir'],
        val_split       = dcfg['val_split'],
        batch_size      = cfg['training']['batch_size'],
        num_workers     = dcfg.get('num_workers', 0),
        noisy_patch_size= dcfg.get('noisy_patch_size', 64),
        clip_sigma      = dcfg.get('clip_sigma', 3.0),
        seed            = seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    mcfg  = cfg['model']
    model = SemiRestoreNet(
        in_ch             = mcfg['in_ch'],
        out_ch            = mcfg['out_ch'],
        base_ch           = mcfg['base_ch'],
        bottleneck_blocks = mcfg['bottleneck_blocks'],
        scale             = mcfg['scale'],
    ).to(device)
    print(f"[Model] SemiRestoreNet  params: {count_params(model)/1e6:.2f}M")

    # ── Loss ──────────────────────────────────────────────────────────────
    lcfg      = cfg['loss']
    criterion = RestorationLoss(
        w_char   = lcfg['w_char'],
        w_ssim   = lcfg['w_ssim'],
        w_edge   = lcfg['w_edge'],
        char_eps = lcfg['char_eps'],
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    ocfg = cfg['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = ocfg['lr'],
        weight_decay = ocfg['weight_decay'],
        betas        = tuple(ocfg['betas']),
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────
    scfg = cfg['scheduler']
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0    = scfg['T_0'],
        T_mult = scfg.get('T_mult', 1),
        eta_min= scfg.get('eta_min', 1e-6),
    )

    scaler = GradScaler('cuda', enabled=amp_enabled)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_ssim   = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_ssim   = ckpt.get('best_ssim', 0.0)
        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}, best_ssim={best_ssim:.4f}")

    # ── CSV logger ────────────────────────────────────────────────────────
    csv_exists = os.path.exists(log_file)
    csv_fh = open(log_file, 'a', newline='')
    csv_writer = csv.DictWriter(csv_fh, fieldnames=[
        'epoch', 'lr', 'train_loss', 'val_loss', 'psnr', 'ssim'
    ])
    if not csv_exists:
        csv_writer.writeheader()

    # ── Early stopping state ───────────────────────────────────────────────
    es_cfg    = cfg['early_stopping']
    es_on     = es_cfg.get('enabled', True)
    es_pat    = es_cfg.get('patience', 30)
    es_delta  = es_cfg.get('min_delta', 1e-4)
    no_improve= 0

    # ── Training loop ─────────────────────────────────────────────────────
    n_epochs    = cfg['training']['epochs']
    log_every   = cfg['logging'].get('log_every', 10)
    val_every   = cfg['logging'].get('val_every', 1)
    vis_every   = cfg['logging'].get('save_vis_every', 10)
    save_every  = cfg['checkpoint'].get('save_every', 10)
    keep_n      = cfg['checkpoint'].get('keep_last_n', 3)

    print(f"\n{'='*60}")
    print(f"  Training for {n_epochs} epochs  |  batch_size={cfg['training']['batch_size']}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch:3d}/{n_epochs}]  lr={cur_lr:.2e}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, amp_enabled, log_every
        )
        scheduler.step()

        # Validate
        val_metrics = {'val_loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}
        vis_data    = None
        if epoch % val_every == 0:
            val_metrics, vis_data = validate(
                model, val_loader, criterion, device, amp_enabled
            )
            psnr = val_metrics['psnr']
            ssim = val_metrics['ssim']
            epoch_t = time.time() - t0
            print(
                f"  ✓ Val  loss={val_metrics['val_loss']:.4f}"
                f"  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}"
                f"  ({epoch_t:.1f}s)"
            )

            # Best model
            if ssim > best_ssim + es_delta:
                best_ssim  = ssim
                no_improve = 0
                best_path  = os.path.join(save_dir, 'best_model.pt')
                torch.save(model.state_dict(), best_path)
                print(f"  ★ New best SSIM={best_ssim:.4f}  → saved to {best_path}")
            else:
                no_improve += 1
                print(f"  No improvement ({no_improve}/{es_pat})")

        # Visualisation
        if vis_data is not None and epoch % vis_every == 0:
            sn, sp, sg = vis_data
            vis_path = os.path.join(vis_dir, f'epoch_{epoch:04d}.png')
            save_comparison_grid(sn, sp, sg, vis_path, n_samples=4)

        # Periodic checkpoint
        if epoch % save_every == 0:
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:04d}.pt')
            torch.save({
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_ssim': best_ssim,
            }, ckpt_path)
            remove_old_checkpoints(save_dir, keep_n)
            print(f"  [Checkpoint] Saved {ckpt_path}")

        # CSV log
        csv_writer.writerow({
            'epoch'     : epoch,
            'lr'        : cur_lr,
            'train_loss': train_metrics['loss'],
            'val_loss'  : val_metrics['val_loss'],
            'psnr'      : val_metrics['psnr'],
            'ssim'      : val_metrics['ssim'],
        })
        csv_fh.flush()

        # Early stopping
        if es_on and no_improve >= es_pat:
            print(f"\n[Early Stop] No SSIM improvement for {es_pat} epochs. Stopping.")
            break

    csv_fh.close()
    print(f"\n[Done] Best validation SSIM = {best_ssim:.4f}")
    print(f"       Best model saved to: {os.path.join(save_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
