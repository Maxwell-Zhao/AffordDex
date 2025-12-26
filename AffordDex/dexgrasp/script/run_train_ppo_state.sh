# CUDA_VISIBLE_DEVICES=2 \
# python train.py \
# --task=ShadowHandGraspP \
# --algo=ppo_pose \
# --seed=0 \
# --rl_device=cuda:0 \
# --sim_device=cuda:0 \
# --logdir=logs/unidexgrasp-pose-afford \
# --headless \

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandGrasp \
--algo=ppo \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/unidexgrasp \
--headless \

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --task=ShadowHandGraspP \
# --algo=ppo \
# --seed=0 \
# --rl_device=cuda:0 \
# --sim_device=cuda:0 \
# --logdir=logs/unidexgrasp-pose \
# --headless \

# --task=ShadowHandGrasp \
# --model_dir=example_model/state_based_model.pt #\ # if you want to finetune on previous model
#--test