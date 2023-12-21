# config_model
CUDA_VISIBLE_DEVICES=0 python -W ignore -u test.py \
    --cover_dir "./images/alaskav2/ALASKA_v2_TIFF_256_COLOR/" \
    --stego_dir "./images/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/" \
    --list_dir "./images/alaska2_list/demo_test.txt" \
    --config_path './configs/model_ala2color256ccmdhill40.config'  \
    --weights "./checkpoints/model_ala2color256ccmdhill40.pth.tar"


