"""
Main evaluation script for Texture-Anything model.
Configures all paths and computes all metrics.
"""
import torch
import json
from pathlib import Path
from metrics.utils import match_image_pairs, load_images_from_folder
from metrics import (
    compute_lpips,
    compute_psnr,
    compute_ssim,
    compute_fid,
    compute_clipiqa,
    compute_brisque
)


# =======================
# CONFIGURATION
# =======================
class Config:
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Image dimensions
    IMAGE_SIZE = (512, 512)
    
    # PATHS FOR 2D TEXTURES (for LPIPS, PSNR, SSIM, FID)
    TEXTURE_GT = "data/test/diffuse"                  # Ground truth textures
    TEXTURE_PRED_SD15 = "results/sd15/textures"       # SD1.5 generated textures
    TEXTURE_PRED_SDXL = "results/sdxl/textures"       # SDXL generated textures
    
    # PATHS FOR 3D RENDERINGS (for CLIPIQA, BRISQUE)
    # These are renderings of the 3D object with texture applied
    RENDER_PRED_SD15 = "results/sd15/renders"         # Renderings with SD1.5 texture
    RENDER_PRED_SDXL = "results/sdxl/renders"         # Renderings with SDXL texture
    
    # CAPTIONS FILE
    CAPTIONS_FILE = "data/test/captions.json"
    
    # Output
    OUTPUT_JSON = "results/metrics_results.json"
    


def evaluate_model(model_name: str, texture_pred_path: str, render_pred_path: str):
    """
    Evaluate a model on all metrics.
    
    Args:
        model_name: Model identifier (e.g., "SD1.5")
        texture_pred_path: Path to generated textures
        render_pred_path: Path to renderings with texture
    
    Returns:
        dict with all results
    """
    results = {'model': model_name}
    
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*60}")
    
    # ====================================
    # 1. METRICS ON 2D TEXTURES (with GT)
    # ====================================
    print(f"\n[1/6] Loading texture pairs...")
    preds, targets = match_image_pairs(
        folder_pred=texture_pred_path,
        folder_gt=Config.TEXTURE_GT,
        size=Config.IMAGE_SIZE
    )
    print(f"  Loaded {preds.shape[0]} texture pairs")
    
    print(f"[2/6] Computing LPIPS...")
    lpips_score = compute_lpips(preds, targets, device=Config.DEVICE)
    results['LPIPS'] = lpips_score
    print(f"  LPIPS: {lpips_score:.4f}")
    
    print(f"[3/6] Computing PSNR...")
    psnr_score = compute_psnr(preds, targets, device=Config.DEVICE)
    results['PSNR'] = psnr_score
    print(f"  PSNR: {psnr_score:.2f} dB")
    
    print(f"[4/6] Computing SSIM...")
    ssim_score = compute_ssim(preds, targets, device=Config.DEVICE)
    results['SSIM'] = ssim_score
    print(f"  SSIM: {ssim_score:.4f}")
    
    # ====================================
    # 2. FID (texture distribution)
    # ====================================
    print(f"[5/6] Computing FID...")
    gt_textures, _ = load_images_from_folder(
        Config.TEXTURE_GT, size=Config.IMAGE_SIZE, normalize=True
    )
    pred_textures, _ = load_images_from_folder(
        texture_pred_path, size=Config.IMAGE_SIZE, normalize=True
    )
    
    fid_score = compute_fid(gt_textures, pred_textures, device=Config.DEVICE)
    results['FID'] = fid_score
    print(f"  FID: {fid_score:.2f}")
    
    # ====================================
    # 3. METRICS ON 3D RENDERINGS
    # ====================================
    print(f"[6/6] Computing metrics on 3D renderings...")
    renders, render_filenames = load_images_from_folder(
        render_pred_path, size=Config.IMAGE_SIZE, normalize=True
    )
    print(f"  Loaded {renders.shape[0]} renderings")
    
    # Load captions
    from metrics.utils import load_captions_json, get_captions_for_filenames
    
    if Path(Config.CAPTIONS_FILE).exists():
        print("  [6a] CLIP-IQA: Prompt fidelity...")
        
        # Load captions dict
        captions_dict = load_captions_json(Config.CAPTIONS_FILE)
        
        # Get captions for render filenames
        captions = get_captions_for_filenames(render_filenames, captions_dict)
        
        # Filter out empty captions
        valid_indices = [i for i, c in enumerate(captions) if c]
        if len(valid_indices) < len(captions):
            print(f"    Warning: {len(captions) - len(valid_indices)} missing captions")
        
        if valid_indices:
            valid_renders = renders[valid_indices]
            valid_captions = [captions[i] for i in valid_indices]
            
            clipiqa_score = compute_clipiqa(
                valid_renders,
                valid_captions,
                device=Config.DEVICE
            )
            results['CLIPIQA'] = clipiqa_score
            print(f"    CLIPIQA: {clipiqa_score:.4f}")
    else:
        print(f"  [6a] Skipping CLIPIQA (captions file not found: {Config.CAPTIONS_FILE})")
    
    # BRISQUE
    print("  [6b] BRISQUE: Artifact assessment...")
    brisque_score = compute_brisque(renders)
    if not torch.isnan(torch.tensor(brisque_score)):
        results['BRISQUE'] = brisque_score
        print(f"    BRISQUE: {brisque_score:.2f}")
    
    return results

def main():
    """Main function."""
    print(f"Device: {Config.DEVICE}")
    print(f"Image size: {Config.IMAGE_SIZE}")
    
    all_results = {}
    
    # Evaluate SD1.5
    results_sd15 = evaluate_model(
        model_name="SD1.5",
        texture_pred_path=Config.TEXTURE_PRED_SD15,
        render_pred_path=Config.RENDER_PRED_SD15
    )
    all_results['SD1.5'] = results_sd15
    
    # Evaluate SDXL
    results_sdxl = evaluate_model(
        model_name="SDXL",
        texture_pred_path=Config.TEXTURE_PRED_SDXL,
        render_pred_path=Config.RENDER_PRED_SDXL
    )
    all_results['SDXL'] = results_sdxl
    
    # Save results
    output_path = Path(Config.OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"{'='*60}\n")
    
    # Print comparison table
    print("\nCOMPARATIVE TABLE:")
    print(f"{'Metric':<15} {'SD1.5':<12} {'SDXL':<12}")
    print("-" * 40)
    for metric in ['LPIPS', 'PSNR', 'SSIM', 'FID', 'BRISQUE']:
        if metric in results_sd15 and metric in results_sdxl:
            print(f"{metric:<15} {results_sd15[metric]:<12.4f} {results_sdxl[metric]:<12.4f}")


if __name__ == "__main__":
    main()
