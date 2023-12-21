# After searching, you can parse the searched architecture
CUDA_VISIBLE_DEVICES=1 python -u parse_s2.py \
    --model_path "/data/liqs/automl/TF-NAS/search-20230206-000830-stage2gc_op5_ala2color256geapcpvs0p40_wisernetboss_custom_adamax_gc5_wmall_pc110/searched_model_best.pth.tar" \
    --save_path "./configs/stage2gc_d3222op5_ala2color256geapcpvs0p40_wisernetboss_pc110_from71.config"
    
# --lookup_path "./latency_pkl/latency_gpu.pkl"
# --stage1_path "/pubdata/liqs/automl/TF-NAS/search-20221009-013049-gcstage2d_ala2color256qf75ccfrju40_d3222op3_custom_adamax_gc5_wmall/searched_model_best.pth.tar" \
