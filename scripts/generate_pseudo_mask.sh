# NEED TO SET
DATASET_ROOT=PATH/TO/DATASET
GPU=0

 1. train a classification network and compute refined seed
 CUDA_VISIBLE_DEVICES=${GPU} python run_sample.py \
     --voc12_root ../data/voc12
     --num_workers 8
     --train_cam_pass True
     --cam_eval_thres 0.15
     --make_cam_pass True
     --eval_cam_pass True
     --cam_to_ir_label_pass True
     --conf_fg_thres 0.30
     --conf_bg_thres 0.05


# 2.1. train an UNS network
 CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample.py \
     --voc12_root ../data/voc12
     --num_workers 8
     --train_uns_pass True

# 2.2. generate activation maps and refined seed for boundary refinement
CUDA_VISIBLE_DEVICES=${GPU} python run_sample.py \
    --voc12_root ../data/voc12\
    --num_workers 8\
    --make_uns_cam_pass True\
    --eval_uns_cam_pass True\
    --uns_cam_to_ir_label_pass True\
    --conf_fg_thres 0.45\
    --conf_bg_thres 0.15

 3.1. train a boundary refinement network (IRN)
 CUDA_VISIBLE_DEVICES=${GPU} python run_sample.py \
     --voc12_root ../data/voc12 \
     --num_workers 4 \
     --train_irn_pass True

# 3.2. generate the final pseudo-masks
CUDA_VISIBLE_DEVICES=${GPU} python run_sample.py \
    --voc12_root ../data/voc12
    --num_workers 8
    --make_sem_seg_pass True
    --eval_sem_seg_pass True
