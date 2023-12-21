# After searching, you can parse the searched architecture
CUDA_VISIBLE_DEVICES=4 python -u parse_s1.py \
    --model_path "./checkpoints/s1_best_ala2color256ccmdhill40.pth.tar" \
    --save_path "./derive_s1.pth.tar"
    