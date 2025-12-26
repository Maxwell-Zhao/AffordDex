# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python train.py \
--task=ShadowImitator \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/imitator \
--headless \
--model_dir=logs/imitator_seed0/model_3100.pt 