import json, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = str(REPO_ROOT)

# 1. Dataset.json
with open(os.path.join(BASE, "data/nnUNet_raw/Dataset003_Combined/dataset.json")) as f:
    ds = json.load(f)
print("=== dataset.json ===")
print("Name:", ds.get("name", "?"))
print("Channel names:", ds.get("channel_names", ds.get("modality", "?")))
print("Labels:", ds.get("labels", "?"))
print("Num training:", ds.get("numTraining", len(ds.get("training", []))))
print("File ending:", ds.get("file_ending", "?"))
print()

# 2. nnUNetPlans (CNN default)
with open(os.path.join(BASE, "data/nnUNet_preprocessed/Dataset003_Combined/nnUNetPlans.json")) as f:
    plans = json.load(f)
cfg = plans["configurations"]["3d_fullres"]
print("=== nnUNetPlans 3d_fullres (CNN) ===")
print("Spacing:", cfg["spacing"])
print("Patch size:", cfg["patch_size"])
print("Batch size:", cfg["batch_size"])
print("Median shape:", cfg.get("median_image_size_in_voxels", "?"))
print("UNet class:", cfg.get("UNet_class_name", "?"))
print("n_conv_per_stage_encoder:", cfg.get("n_conv_per_stage_encoder", "?"))
print("n_conv_per_stage_decoder:", cfg.get("n_conv_per_stage_decoder", "?"))
print("pool_op_kernel_sizes:", cfg.get("pool_op_kernel_sizes", "?"))
print("UNet_base_num_features:", cfg.get("UNet_base_num_features", "?"))
print("num_pool_per_axis:", cfg.get("num_pool_per_axis", "?"))
print()

# 3. ResEncL plans for comparison
resencl_path = os.path.join(BASE, "data/nnUNet_preprocessed/Dataset003_Combined/nnUNetResEncUNetLPlans.json")
if os.path.exists(resencl_path):
    with open(resencl_path) as f:
        rplans = json.load(f)
    rcfg = rplans["configurations"]["3d_fullres"]
    print("=== nnUNetResEncUNetLPlans 3d_fullres (ResEncL) — for comparison ===")
    print("Spacing:", rcfg["spacing"])
    print("Patch size:", rcfg["patch_size"])
    print("Batch size:", rcfg["batch_size"])
    print("UNet class:", rcfg.get("UNet_class_name", "?"))
    print()

# 4. Dataset fingerprint
fp_path = os.path.join(BASE, "data/nnUNet_preprocessed/Dataset003_Combined/dataset_fingerprint.json")
if os.path.exists(fp_path):
    with open(fp_path) as f:
        fp = json.load(f)
    print("=== Dataset Fingerprint ===")
    spacings = fp.get("spacings", [])
    if spacings:
        import numpy as np
        sp_arr = np.array(spacings)
        print("Num cases:", len(spacings))
        print("Spacing min:", sp_arr.min(axis=0).tolist())
        print("Spacing median:", np.median(sp_arr, axis=0).tolist())
        print("Spacing max:", sp_arr.max(axis=0).tolist())
    
    fip = fp.get("foreground_intensity_properties_per_channel", {})
    for ch, props in fip.items():
        print(f"  Channel {ch}: mean={props.get('mean', '?'):.2f}, std={props.get('std', '?'):.2f}, "
              f"p0.5={props.get('percentile_00_5', '?'):.2f}, p99.5={props.get('percentile_99_5', '?'):.2f}")
    print()

# 5. Check normalization schemes in plans
print("=== Normalization ===")
for ch_key, scheme in plans.get("normalization_schemes", {}).items():
    print(f"  Channel {ch_key}: {scheme}")
print()

# 6. Verify a sample preprocessed case
import blosc2
sample_dir = os.path.join(BASE, "data/nnUNet_preprocessed/Dataset003_Combined/nnUNetPlans_3d_fullres")
samples = sorted([f for f in os.listdir(sample_dir) if f.endswith(".b2nd")])[:2]
for s in samples:
    arr = blosc2.open(os.path.join(sample_dir, s))
    print(f"  {s}: shape={arr.shape}, dtype={arr.dtype}")
