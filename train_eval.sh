# !/usr/local/bin sh

CUDA_VISIBLE_DEVICES=1 python -u train_eval.py \
    --cover_dir "./images/alaskav2/ALASKA_v2_TIFF_256_COLOR/" \
    --stego_dir "./images/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/" \
    --list_dir "./images/alaska2_list/" \
    --config_path "./configs/model_ala2color256ccmdhill40.config"  \
    --resume_derived_model "./checkpoints/derive_s2_ala2color256ccmdhill40.pth.tar"  \
    --save "./stars_logs/" \
    --note "demo_train"

