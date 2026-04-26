"""
main.py —
"""     
import sys
from pathlib import Path

# Windows + IPython fix: multiprocessing spawn needs __spec__
if '__main__' in sys.modules:
    sys.modules['__main__'].__spec__ = getattr(
        sys.modules['__main__'], '__spec__', None)

import torch
import clip

from config   import parse_args
from datasets import build_dataloader
from utils    import (OUT, select_device, run_zeroshot, load_previous_results,
                      plot_comparison, plot_cocoop_curves,
                      save_results, merge_all_csvs, print_results_table)
from engine   import train_cocoop, hp_sweep



def main():
    args   = parse_args()
    device = select_device(args.cuda_device)
    print(f'Dataset : {args.dataset}  |  Backbone : {args.backbone}\n')

    if args.merge_csv:
        merge_all_csvs(); return

    # ── Load CLIP backbone ─────────────────────────────────────
    clip_model, _ = clip.load(args.backbone, device=device)
    clip_model.eval()

    test_loader, class_names = build_dataloader(
        args.dataset, args.data_root, split='test',
        batch_size=args.batch_size, num_workers=args.num_workers)
    print(f'Classes : {len(class_names)}  |  test batches : {len(test_loader)}')


    results = load_previous_results(args.dataset)

    # ── Zero-shot baselines ────────────────────────────────────
    if not args.skip_zeroshot:
        zs = run_zeroshot(clip_model, clip.tokenize,
                          test_loader, class_names, args.dataset, device)
        results['ZS-handcrafted'] = zs['zs_handcrafted']
        results['ZS-LLM']         = zs['zs_llm']
    else:
        print('\n[Zero-shot] skipped — using cached baselines.')

    # ── CoCoOp ────────────────────────────────────────────────
    if not args.skip_cocoop:
        print('\n[CoCoOp] image-conditioned dynamic prompts …')
        train_loader, _ = build_dataloader(
            args.dataset, args.data_root, split='train',
            batch_size=args.batch_size, num_workers=args.num_workers,
            shots=args.shots, train=True)

        model_cocoop, hist_cocoop = train_cocoop(
            clip_model, class_names, train_loader, test_loader, device,
            n_ctx=args.n_ctx, lr=args.lr, epochs=args.epochs,
            hidden_ratio=args.hidden_ratio, eval_freq=args.eval_freq)

        plot_cocoop_curves(hist_cocoop, args.dataset)
        results['CoCoOp'] = {'top1': hist_cocoop[-1]['top1'],
                             'top5': hist_cocoop[-1]['top5']}
        torch.save(model_cocoop.state_dict(),
                   OUT / f'cocoop_weights_{args.dataset}.pth')
        print(f'  weights saved → results/cocoop_weights_{args.dataset}.pth')

    # ── HP sweep ──────────────────────────────────────────────
    if args.hp_sweep:
        print('\n[HP Sweep] …')
        sweep_loader, _ = build_dataloader(
            args.dataset, args.data_root, split='train',
            batch_size=args.batch_size, num_workers=args.num_workers,
            shots=args.shots, train=True)
        hp_sweep(clip_model, class_names, sweep_loader, test_loader, device)

    # ── Attention heatmap ─────────────────────────────────────
    if args.visualize:
        import os
        from visualize import get_text_to_image_heatmaps, render_comparison
        if not os.path.exists(args.vis_image):
            print(f'\n[Visualize] image not found: {args.vis_image}')
        else:
            print(f'\n[Visualize] {args.dataset} class {args.vis_class_idx} …')
            raw_img, zs_heat, cocoop_heat, name = get_text_to_image_heatmaps(
                args.dataset, args.vis_class_idx, args.vis_image, device)
            render_comparison(raw_img, zs_heat, cocoop_heat, name,
                              str(OUT / f'heatmap_{args.dataset}.png'))

    # ── Save + print ───────────────────────────────────────────
    plot_comparison(results, args.dataset)
    save_results(results, args.dataset)
    print_results_table(results)


if __name__ == '__main__':
    main()
