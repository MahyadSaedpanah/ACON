 
CUDA_VISIBLE_DEVICES=0 python main.py \
 --experiment_description ACON \
 --run_description WISDM \
 --da_method ACON \
 --dataset WISDM \
 --num_runs 5 \
 --lr 0.003 \
 --cls_trade_off 1 \
 --domain_trade_off 1 \
 --entropy_trade_off 0.02 \
 --align_t_trade_off 2 \
 --align_s_trade_off 1 \
 --kl_t 3.0 \
 --uncertainty_weight 0.2 \
 --cl_trade_off 0.01 \
 --cl_conf_thresh 0.80 \
 --cl_temp 0.3 \
 --debug \
#  --device cuda \