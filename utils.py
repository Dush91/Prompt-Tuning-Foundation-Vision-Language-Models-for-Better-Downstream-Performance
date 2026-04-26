"""
utils.py — model-agnostic helpers.
Knows nothing about CoOp / CoCoOp / KgCoOp architectures.
"""
import random, json, csv, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import CLIP_TEMPLATES, LLM_PROMPTS, build_zeroshot_weights

OUT = Path('results')
OUT.mkdir(exist_ok=True)


# ── Reproducibility ───────────────────────────────────────────
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── Metrics ───────────────────────────────────────────────────
def topk_acc(logits, labels, k=5):
    k = min(k, logits.shape[1])
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return (correct[:, 0].float().mean().item() * 100,
            correct.any(dim=1).float().mean().item() * 100)


# ── Device ────────────────────────────────────────────────────
def select_device(cuda_device=0):
    if not torch.cuda.is_available():
        print('Device : cpu'); return 'cpu'

    count = torch.cuda.device_count()
    if cuda_device >= count:
        raise SystemExit(
            f'cuda:{cuda_device} not found. '
            f'Available: {[torch.cuda.get_device_name(i) for i in range(count)]}')

    torch.cuda.set_device(cuda_device)
    device = f'cuda:{cuda_device}'
    vram = torch.cuda.get_device_properties(cuda_device).total_memory / 1e9
    print(f'Device : {device}  |  {torch.cuda.get_device_name(cuda_device)}  ({vram:.1f} GB)')
    torch.backends.cudnn.benchmark = True
    return device


# ── Persistence ───────────────────────────────────────────────
def save_results(results, dataset_name):
    json_path = OUT / f'results_{dataset_name}.json'
    csv_path  = OUT / f'results_{dataset_name}.csv'

    with open(json_path, 'w') as f:
        json.dump({'dataset': dataset_name, 'results': results}, f, indent=2)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'method', 'top1', 'top5'])
        for method, vals in results.items():
            w.writerow([dataset_name, method,
                        round(vals['top1'], 2), round(vals.get('top5', 0), 2)])
    print(f'  saved → {json_path}  |  {csv_path}')


def load_previous_results(dataset_name):
    p = OUT / f'results_{dataset_name}.json'
    if not p.exists():
        print(f'  Warning: {p} not found — run without --skip_zeroshot first!')
        return {}
    with open(p) as f:
        return json.load(f).get('results', {})


def merge_all_csvs():
    csvs = list(OUT.glob('results_*.csv'))
    if not csvs: return
    rows, header = [], None
    for c in sorted(csvs):
        with open(c) as f:
            r = csv.reader(f); h = next(r)
            if header is None: header = h
            rows.extend(r)
    merged = OUT / 'results_all.csv'
    with open(merged, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f'  merged → {merged}')


# ── Zero-shot pipeline ────────────────────────────────────────
def cache_image_features(clip_model, loader, device):
    """Encode all test images once; keep on CPU to save VRAM."""
    feats, labels_all = [], []
    clip_model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='  caching images'):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                f = clip_model.encode_image(imgs).float()
            feats.append(F.normalize(f, dim=-1).cpu())
            labels_all.append(labels)
    return torch.cat(feats), torch.cat(labels_all)


def eval_with_weights(feats, labels, weights):
    logits = feats @ weights * 100.0
    t1 = (logits.argmax(1) == labels).float().mean().item() * 100
    k  = min(5, logits.shape[1])
    t5 = (logits.topk(k, dim=1).indices ==
          labels.unsqueeze(1)).any(1).float().mean().item() * 100
    return t1, t5


def run_zeroshot(clip_model, tokenizer, test_loader,
                 class_names, dataset_name, device):
    print('\n[Zero-shot] caching image features …')
    feats, labels = cache_image_features(clip_model, test_loader, device)

    print('[Zero-shot] evaluating 80 templates …')
    per_tmpl = [
        eval_with_weights(
            feats, labels,
            build_zeroshot_weights(clip_model, tokenizer,
                                   class_names, [t], device).cpu())[0]
        for t in tqdm(CLIP_TEMPLATES, desc='  templates')
    ]
    accs = np.array(per_tmpl)

    w_ens = build_zeroshot_weights(clip_model, tokenizer,
                                   class_names, CLIP_TEMPLATES, device).cpu()
    ens_t1, ens_t5 = eval_with_weights(feats, labels, w_ens)

    llm_tmpls = LLM_PROMPTS.get(dataset_name, CLIP_TEMPLATES[:6])
    w_llm = build_zeroshot_weights(clip_model, tokenizer,
                                   class_names, llm_tmpls, device).cpu()
    llm_t1, llm_t5 = eval_with_weights(feats, labels, w_llm)

    print(f'  ensemble  top1={ens_t1:.2f}%  top5={ens_t5:.2f}%')
    print(f'  LLM       top1={llm_t1:.2f}%  top5={llm_t5:.2f}%')

    plot_zeroshot(accs, ens_t1, llm_t1, dataset_name)
    return {
        'per_template':   accs,
        'zs_handcrafted': {'top1': ens_t1, 'top5': ens_t5},
        'zs_llm':         {'top1': llm_t1, 'top5': llm_t5},
    }


# ── Plots ─────────────────────────────────────────────────────
def _save(name):
    p = OUT / name
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f'  saved → {p}')


def plot_zeroshot(accs, ens_t1, llm_t1, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.bar(range(len(accs)), sorted(accs, reverse=True), color='#4C72B0', alpha=0.8)
    ax.axhline(accs.mean(), color='red', lw=2, label=f'Mean={accs.mean():.1f}%')
    ax.fill_between(range(len(accs)),
                    accs.mean()-accs.std(), accs.mean()+accs.std(),
                    alpha=0.15, color='red', label=f'±σ={accs.std():.1f}%')
    ax.scatter([0], [ens_t1], color='orange', zorder=5, s=80,
               marker='D', label=f'Ensemble={ens_t1:.1f}%')
    ax.set(xlabel='Template rank', ylabel='Top-1 (%)',
           title=f'Zero-shot variance — {dataset_name}')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    parts = ax2.violinplot([accs], positions=[1], widths=0.6,
                           showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#4C72B0'); pc.set_alpha(0.6)
    ax2.scatter([1], [ens_t1], color='orange', zorder=5, s=80,
                marker='D', label=f'Ensemble={ens_t1:.1f}%')
    ax2.scatter([1], [llm_t1], color='green',  zorder=5, s=80,
                marker='*', label=f'LLM={llm_t1:.1f}%')
    ax2.set_xticks([1]); ax2.set_xticklabels([dataset_name])
    ax2.set(ylabel='Top-1 (%)', title='Accuracy distribution')
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(f'zeroshot_{dataset_name}.png')


def _plot_curves(history, dataset_name, label, loss_color, acc_color):
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h['loss'] for h in history], 'o-', color=loss_color, lw=2)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title=f'{label} training loss')
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [h['top1'] for h in history], 's-', color=acc_color, lw=2)
    axes[1].set(xlabel='Epoch', ylabel='Top-1 (%)', title=f'{label} test accuracy')
    axes[1].grid(alpha=0.3)
    plt.suptitle(f'{label} — {dataset_name}')
    plt.tight_layout()
    _save(f'{label.lower()}_curves_{dataset_name}.png')


def plot_cocoop_curves(h, ds): _plot_curves(h, ds, 'CoCoOp', '#C44E52', '#55A868')


def plot_comparison(results, dataset_name):
    methods = list(results.keys())
    top1s   = [results[m]['top1'] for m in methods]
    top5s   = [results[m]['top5'] for m in methods]
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'][:len(methods)]

    x, w = np.arange(len(methods)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, top1s, w, label='Top-1', color=colors, alpha=0.9)
    ax.bar(x + w/2, top5s, w, label='Top-5', color=colors, alpha=0.45)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set(ylabel='Accuracy (%)', ylim=(0, 105),
           title=f'Method comparison — {dataset_name}')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(f'comparison_{dataset_name}.png')


def plot_hp_sensitivity(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for col, hp in enumerate(['n_ctx', 'lr', 'hidden_ratio']):
        ax   = axes[col]
        vals = sorted(set(r[hp] for r in results))
        means= [np.mean([r['top1'] for r in results if r[hp]==v]) for v in vals]
        ax.bar(range(len(vals)), means, color='#4C72B0', alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([str(v) for v in vals])
        ax.set(xlabel=hp, ylabel='Mean top-1 (%)', title=f'Sensitivity: {hp}')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save('hp_sensitivity.png')


def print_results_table(results):
    print(f"\n{'Method':<20} {'Top-1':>8} {'Top-5':>8}")
    print('─' * 40)
    for m, v in results.items():
        print(f"{m:<20} {v['top1']:>7.2f}% {v.get('top5',0):>7.2f}%")
