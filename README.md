# This is the supplementary materials of STARS

-----------------------------------
### Requirements

* python 3.7
* pytorch 1.9.0
---
### Take the following an example on ALASKA-v2 dataset with CMD-C-HILL 0.4 bpc



#### Search

1. For searching the preprocessing layer in the first stage :   (./search_s1.sh)

    ```sh
    CUDA_VISIBLE_DEVICES=4 python -u search_s1.py \
        --cover_dir "./images/alaskav2/ALASKA_v2_TIFF_256_COLOR/" \
        --stego_dir "./images/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/" \
        --list_dir "./images/alaska2_list/" \
        --save "./stars_logs/"  \
        --note "demo_s1"
    ```

2. Derive the preprocessing layer in the first stage: (parse_s1.sh)

   ```sh
   CUDA_VISIBLE_DEVICES=0 python -u parse_s1.py \
       --model_path  "./checkpoints/s1_best_ala2color256ccmdhill40.pth.tar" \
       --save_path "./derive_s1.pth.tar"
   ```

3. Search the steganalytic architecture: (search_s2.sh)

    ```sh
    CUDA_VISIBLE_DEVICES=5 python -W ignore -u search_s2.py \
        --cover_dir "./images/alaskav2/ALASKA_v2_TIFF_256_COLOR/" \
        --stego_dir "./images/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/" \
        --list_dir "./images/alaska2_list/" \
        --save "./stars_logs/" \
        --note "demo_s2" \
        --resume "./checkpoints/derive_s1_ala2color256ccmdhill40.pth.tar"
    ```


4. Derive the final architecture: (parse_s2.sh)

    ```sh
    CUDA_VISIBLE_DEVICES=0 python -u parse_s2.py \
        --model_path "./checkpoints/s2_best_ala2color256ccmdhill40.pth.tar" \
        --save_model "./derive_s2.pth.tar"  \
        --save_cfg "./parse_s2.config"
    ```

    You will get a model config file and corresponding checkpoint for training and testing, as well as some model profile information.



#### Train

Train the derived architecture: (train_eval.sh)

```sh
CUDA_VISIBLE_DEVICES=1 python -u train_eval.py \
    --cover_dir "./images/alaskav2/ALASKA_v2_TIFF_256_COLOR/" \
    --stego_dir "./images/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/" \
    --list_dir "./images/alaska2_list/" \
    --config_path "./configs/model_ala2color256ccmdhill40.config"  \
    --resume_derived_model "./checkpoints/derive_s2_ala2color256ccmdhill40.pth.tar"  \
    --save "./stars_logs/" \
    --note "demo_train"
```



#### Test

After training, you can test the trained model by:  (test.sh)

```sh
CUDA_VISIBLE_DEVICES=0 python -W ignore -u test.py \
    --cover_dir "./images/alaskav2/ALASKA_v2_TIFF_256_COLOR/" \
    --stego_dir "./images/alaskav2/color_spatial/COLOR256_HILL_CCMD_40/" \
    --list_dir "./images/alaska2_list/demo_test.txt"  \
    --config_path './configs/model_ala2color256ccmdhill40.config'  \
    --weights "./checkpoints/model_ala2color256ccmdhill40.pth.tar"
```



___
### Checkpoint description
'~/srnet_tensor_decomposition/official'

Checkpoint derived from the first stage of STARS to be searched in the second stage of STARS: 
    './checkpoints/derive_s1_ala2color256ccmdhill40.pth.tar'

Checkpoint derived from the second stage of STARS to be trained: 
    './checkpoints/derive_s2_ala2color256ccmdhill40.pth.tar'
    and the corresponding config file: './configs/model_ala2color256ccmdhill40.config'
    
The final steganalytic architecture to be evaluated:
    './checkpoints/model_ala2color256ccmdhill40.pth.tar'
