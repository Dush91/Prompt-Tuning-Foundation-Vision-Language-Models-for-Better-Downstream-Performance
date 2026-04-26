"""
visualize.py — Prompt-Conditioned Attention Maps
"""
import os, torch, clip
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path

# Import your custom architecture
from datasets import DATASETS
from models import CoCoOp, build_zeroshot_weights, LLM_PROMPTS

def get_text_to_image_heatmaps(dataset_name, target_class_idx, image_path, device="cuda"):
    print(f"Loading ViT-B/16 and {dataset_name} architecture...")
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    
    # 1. Setup Dataset & Model
    ds_class = DATASETS[dataset_name]
    temp_ds = ds_class(root='./data', split='test', download=False)
    class_names = temp_ds.class_names
    target_class_name = class_names[target_class_idx]
    
    model = CoCoOp(clip_model, class_names, n_ctx=16, hidden_ratio=0.125).to(device)
    
    # Load your trained weights from the results folder!
    weight_path = Path('results') / f"cocoop_weights_{dataset_name}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing {weight_path}. Run your training cell for {dataset_name} first!")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

# 2. Hook to extract dense image patches from the Vision Transformer
    patch_features = None
    def hook_fn(module, input, output):
        nonlocal patch_features
        
        patches = output[1:, 0, :]                   
        patches = clip_model.visual.ln_post(patches) 
        proj = clip_model.visual.proj                
        patches = patches @ proj                     
        patch_features = F.normalize(patches, dim=-1)

    
    handle = clip_model.visual.transformer.register_forward_hook(hook_fn)

    # 3. Process Image
    print(f"Processing image for class: '{target_class_name}'")
    raw_image = Image.open(image_path).convert("RGB")
    image_input = preprocess(raw_image).unsqueeze(0).to(device)

    # 4. Trigger Forward Pass
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            img_feat = F.normalize(clip_model.encode_image(image_input), dim=-1)
            
            # --- Get CoCoOp Prompt Vector ---
            delta = model.meta_net(img_feat)
            ctx_shifted = model.ctx.unsqueeze(0) + delta.unsqueeze(1)
            all_cocoop_feats = model._encode_prompts_batch(ctx_shifted) 
            cocoop_target_feat = all_cocoop_feats[0, target_class_idx, :] 
            
            # --- Get Zero-Shot Prompt Vector ---
            tmpl = LLM_PROMPTS.get(dataset_name, ["a photo of a {}."])[0]
            zs_all_feats = build_zeroshot_weights(clip_model, clip.tokenize, class_names, [tmpl], device)
            zs_target_feat = zs_all_feats[:, target_class_idx] 

    handle.remove()

    # 5. Calculate Cosine Similarity (The Attention Map)
    zs_sim = (patch_features @ zs_target_feat).cpu().numpy().reshape(14, 14)
    cocoop_sim = (patch_features @ cocoop_target_feat).cpu().numpy().reshape(14, 14)

    # Normalize between 0 and 1
    zs_sim = (zs_sim - zs_sim.min()) / (zs_sim.max() - zs_sim.min() + 1e-8)
    cocoop_sim = (cocoop_sim - cocoop_sim.min()) / (cocoop_sim.max() - cocoop_sim.min() + 1e-8)

    return raw_image, zs_sim, cocoop_sim, target_class_name

def render_comparison(raw_img, zs_heat, cocoop_heat, class_name, save_path="results/comparison_heatmap.png"):
    img = np.array(raw_img.resize((224, 224)))
    
    def apply_heatmap(heat):
        h_resized = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_CUBIC)
        h_colored = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
        h_colored = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img, 0.4, h_colored, 0.6, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\n(Target: {class_name})", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(apply_heatmap(zs_heat))
    axes[1].set_title("Zero-Shot Attention", fontsize=14, color='#4C72B0')
    axes[1].axis('off')
    
    axes[2].imshow(apply_heatmap(cocoop_heat))
    axes[2].set_title("CoCoOp Attention", fontsize=14, color='#C44E52')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nSaved! Open {save_path} to see the results.")

if __name__ == "__main__":
    # --- SETUP YOUR TEST HERE ---
    DATASET = "dtd"          # Make sure you've trained this dataset!
    TARGET_CLASS_INDEX = 0  # The index number of the class in the image
    TEST_IMAGE = "sample.jpg" 
    
    if os.path.exists(TEST_IMAGE):
        raw_img, zs, cocoop, name = get_text_to_image_heatmaps(DATASET, TARGET_CLASS_INDEX, TEST_IMAGE)
        render_comparison(raw_img, zs, cocoop, name, f"results/heatmap_{DATASET}.png")
    else:
        print(f"Error: Put an image named '{TEST_IMAGE}' in your folder to test!")