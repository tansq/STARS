
CUDA_VISIBLE_DEVICES=5 python -W ignore -u search_s2.py \
    --cover_dir "./images/alaskav2/ALASKA_v2_TIFF_256_COLOR/" \
    --stego_dir "./images/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/" \
    --list_dir "./images/alaska2_list/" \
    --save "./stars_logs/" \
    --note "demo_s2" \
    --resume "./checkpoints/derive_s1_ala2color256ccmdhill40.pth.tar"

# /pubdata/liqs/datasets/alaskav2/ALASKA_v2_TIFF_256_COLOR/
# /pubdata/liqs/datasets/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/

# /pubdata/liqs/datasets/alaskav2/color_noround_ycrcb/ALASKA_v2_JPG_256_QF75_COLOR_NR_YCRCB
# COLOR256_QF75_CCFR_JUNIWARD20_NR_YCRCB
