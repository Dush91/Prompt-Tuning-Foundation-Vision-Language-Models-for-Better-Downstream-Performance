"""
engine.py — all training loops and evaluation.
Rule: one eval function, three trainers, one hp sweeper.
CoCoOp uses cached features (fast). CoOp + KgCoOp use raw loader (no forward_cached).
"""
import os, json, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import clip

from models import CoCoOp, build_zeroshot_weights
from utils  import OUT, set_seed, topk_acc, plot_hp_sensitivity


# ── LR scheduler ─────────────────────────────────────────────
def cosine_lr(optimizer, base_lr, warmup_steps, total_steps, step):
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        e = step - warmup_steps; es = total_steps - warmup_steps
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * e / max(1, es)))
    for g in optimizer.param_groups: g['lr'] = lr


# ── Unified evaluator ────────────────────────────────────────
@torch.no_grad()
def eval_model(model, loader, device, dual_output=False,
               cached_feats=None, cached_labels=None):
    """
    fast path (CoCoOp only): pass cached_feats + cached_labels → forward_cached
    slow path (CoOp, KgCoOp): pass raw DataLoader → model(imgs)
    dual_output=True for KgCoOp which returns (logits, text_feats)
    """
    model.eval()
    t1s = t5s = n = 0

    use_cache = (cached_feats is not None
                 and cached_labels is not None
                 and hasattr(model, 'forward_cached'))

    if use_cache:
        for i in range(0, cached_feats.shape[0], 32):
            f = cached_feats[i:i+32]; l = cached_labels[i:i+32]
            with torch.amp.autocast('cuda'):
                out = model.forward_cached(f).float()
            t1, t5 = topk_acc(out, l)
            t1s += t1*l.size(0); t5s += t5*l.size(0); n += l.size(0)
    else:
        for imgs, labels in tqdm(loader, desc='  eval', leave=False):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
            logits = out[0] if dual_output else out
            logits = logits.float()
            t1, t5 = topk_acc(logits, labels)
            t1s += t1*labels.size(0); t5s += t5*labels.size(0); n += labels.size(0)

    return t1s / n, t5s / n


# ── Shared: cache features from frozen CLIP encoder ──────────
@torch.no_grad()
def _cache_features(clip_encoder, loader, device, desc='caching'):
    clip_encoder.eval()
    feats, labels_all = [], []
    for imgs, labels in tqdm(loader, desc=f'  {desc}', leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            f = clip_encoder.encode_image(imgs).float()
        feats.append(F.normalize(f, dim=-1))
        labels_all.append(labels.to(device, non_blocking=True))
    return torch.cat(feats), torch.cat(labels_all)


def _cached_loader(feats, labels, batch_size):
    return DataLoader(TensorDataset(feats, labels),
                      batch_size=batch_size, shuffle=True)


# ── CoCoOp ───────────────────────────────────────────────────
def train_cocoop(clip_model, class_names, train_loader, test_loader,
                 device, n_ctx=16, lr=2e-3, epochs=20,
                 hidden_ratio=0.125, eval_freq=10):
    set_seed()
    model  = CoCoOp(clip_model, class_names,
                    n_ctx=n_ctx, hidden_ratio=hidden_ratio).to(device)

    if hasattr(torch, 'compile') and os.name != 'nt':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('  torch.compile enabled')
        except Exception as e:
            print(f'  torch.compile skipped: {e}')

    params = [p for p in model.parameters() if p.requires_grad]
    opt    = SGD(params, lr=lr, weight_decay=5e-4, momentum=0.9)
    scaler = torch.amp.GradScaler('cuda')

    # Cache both sets — frozen encoder never runs inside loop
    print('  [CoCoOp] caching test features …')
    test_feats, test_labels = _cache_features(
        model.clip, test_loader, device, 'test')
    print('  [CoCoOp] caching train features …')
    train_feats, train_labels = _cache_features(
        model.clip, train_loader, device, 'train')
    print(f'  train={train_feats.shape[0]}  test={test_feats.shape[0]}')

    cached = _cached_loader(train_feats, train_labels, train_loader.batch_size)
    total  = epochs * len(cached); warmup = len(cached)
    history, step = [], 0

    for epoch in range(1, epochs + 1):
        model.train(); epoch_loss = 0.0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(cached, desc=f'  [CoCoOp] {epoch:3d}/{epochs}', leave=False)

        for feats, labels in pbar:
            cosine_lr(opt, lr, warmup, total, step)
            with torch.amp.autocast('cuda'):
                loss = F.cross_entropy(model.forward_cached(feats), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            epoch_loss += loss.item(); step += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        if epoch % eval_freq == 0 or epoch == epochs:
            t1, t5 = eval_model(model, None, device,
                                 cached_feats=test_feats,
                                 cached_labels=test_labels)
            rec = {'epoch': epoch, 'loss': epoch_loss/len(cached),
                   'top1': t1, 'top5': t5}
            history.append(rec)
            print(f'  [CoCoOp] epoch {epoch:3d}  loss={rec["loss"]:.4f}  top1={t1:.2f}%')

    return model, history


# ── HP sweep (CoCoOp only) ────────────────────────────────────
def hp_sweep(clip_model, class_names, train_loader, test_loader, device):
    grid = {'n_ctx': [4, 8, 16], 'lr': [1e-3, 2e-3, 5e-3], 'hidden_ratio': [0.125, 0.25]}
    results = []
    for n_ctx in grid['n_ctx']:
        for lr in grid['lr']:
            for hr in grid['hidden_ratio']:
                print(f'\n[HP] n_ctx={n_ctx}  lr={lr}  hidden={hr}')
                _, hist = train_cocoop(clip_model, class_names,
                                       train_loader, test_loader, device,
                                       n_ctx=n_ctx, lr=lr, hidden_ratio=hr,
                                       epochs=10, eval_freq=10)
                best = max(h['top1'] for h in hist)
                results.append({'n_ctx': n_ctx, 'lr': lr,
                                'hidden_ratio': hr, 'top1': best})
                print(f'  best={best:.2f}%')

    results.sort(key=lambda x: x['top1'], reverse=True)
    for r in results[:3]: print(f'  {r}')
    with open(OUT / 'hp_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    plot_hp_sensitivity(results)
    return results
