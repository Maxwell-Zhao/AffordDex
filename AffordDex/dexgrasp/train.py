# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex

def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["ppo", "dagger", "dagger_value", "ppo_pose"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        # sarl <algorithms.rl.ppo.ppo.PPO object at 0x7f9069c84a30>
        sarl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)
        # print('sarl',sarl)
        # import pdb; pdb.set_trace()
        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    else:
        print("Unrecognized algorithm!")

# bash script/run_train_ppo_state.sh 
# cfg_train {'seed': 0, 'clip_observations': 5.0, 'clip_actions': 1.0, 'policy': {'pi_hid_sizes': [1024, 1024, 512, 512], 'vf_hid_sizes': [1024, 1024, 512, 512], 'activation': 'elu', 'backbone_type': None, 'freeze_backbone': False}, 'learn': {'agent_name': 'shadow_hand', 'test': False, 'resume': 0, 'save_interval': 500, 'print_log': True, 'max_iterations': 10000, 'cliprange': 0.2, 'ent_coef': 0, 'nsteps': 8, 'noptepochs': 5, 'nminibatches': 4, 'max_grad_norm': 1, 'optim_stepsize': 0.0003, 'schedule': 'adaptive', 'desired_kl': 0.016, 'gamma': 0.96, 'lam': 0.95, 'init_noise_std': 0.8, 'log_interval': 1, 'asymmetric': False}}
# cfg {'graphics_device_id': 0, 'env': {'env_name': 'shadow_hand_grasp', 'numEnvs': 1000, 'envSpacing': 1.5, 'episodeLength': 200, 'enableDebugVis': False, 'aggregateMode': 1, 'random_prior': True, 'random_time': True, 'repose_z': True, 'goal_cond': False, 'object_code_dict': {'sem/Car-669043a8ce40d9d78781f76a6db4ab62': [0.06]}, 'stiffnessScale': 1.0, 'forceLimitScale': 1.0, 'useRelativeControl': False, 'dofSpeedScale': 20.0, 'actionsMovingAverage': 1.0, 'controlFrequencyInv': 1, 'startPositionNoise': 0.0, 'startRotationNoise': 0.0, 'resetPositionNoise': 0.0, 'resetRotationNoise': 0.0, 'resetDofPosRandomInterval': 0.0, 'resetDofVelRandomInterval': 0.0, 'distRewardScale': 20, 'transition_scale': 0.5, 'orientation_scale': 0.1, 'rotRewardScale': 1.0, 'rotEps': 0.1, 'actionPenaltyScale': -0.0002, 'reachGoalBonus': 250, 'fallDistance': 0.4, 'fallPenalty': 0.0, 'objectType': 'pot', 'observationType': 'full_state', 'handAgentIndex': '[[0, 1, 2, 3, 4, 5]]', 'asymmetric_observations': False, 'successTolerance': 0.1, 'printNumSuccesses': False, 'maxConsecutiveSuccesses': 0, 'asset': {'assetRoot': '../assets', 'assetFileName': 'mjcf/open_ai_assets/hand/shadow_hand.xml', 'assetFileNameBlock': 'urdf/objects/cube_multicolor.urdf', 'assetFileNameBall': 'urdf/objects/ball.urdf', 'assetFileNameEgg': 'mjcf/open_ai_assets/hand/egg.xml', 'assetFileNamePen': 'mjcf/open_ai_assets/hand/pen.xml'}}, 'task': {'randomize': False, 'randomization_params': {'frequency': 600, 'observations': {'range': [0, 0.002], 'range_correlated': [0, 0.001], 'operation': 'additive', 'distribution': 'gaussian', 'schedule': 'linear', 'schedule_steps': 40000}, 'actions': {'range': [0.0, 0.05], 'range_correlated': [0, 0.015], 'operation': 'additive', 'distribution': 'gaussian', 'schedule': 'linear', 'schedule_steps': 40000}, 'sim_params': {'gravity': {'range': [0, 0.4], 'operation': 'additive', 'distribution': 'gaussian', 'schedule': 'linear', 'schedule_steps': 40000}}, 'actor_params': {'hand': {'color': True, 'tendon_properties': {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform', 'schedule': 'linear', 'schedule_steps': 30000}, 'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform', 'schedule': 'linear', 'schedule_steps': 30000}}, 'dof_properties': {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform', 'schedule': 'linear', 'schedule_steps': 30000}, 'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform', 'schedule': 'linear', 'schedule_steps': 30000}, 'lower': {'range': [0, 0.01], 'operation': 'additive', 'distribution': 'gaussian', 'schedule': 'linear', 'schedule_steps': 30000}, 'upper': {'range': [0, 0.01], 'operation': 'additive', 'distribution': 'gaussian', 'schedule': 'linear', 'schedule_steps': 30000}}, 'rigid_body_properties': {'mass': {'range': [0.5, 1.5], 'operation': 'scaling', 'distribution': 'uniform', 'schedule': 'linear', 'schedule_steps': 30000}}, 'rigid_shape_properties': {'friction': {'num_buckets': 250, 'range': [0.7, 1.3], 'operation': 'scaling', 'distribution': 'uniform', 'schedule': 'linear', 'schedule_steps': 30000}}}, 'object': {'scale': {'range': [0.95, 1.05], 'operation': 'scaling', 'distribution': 'uniform', 'schedule': 'linear', 'schedule_steps': 30000}, 'rigid_body_properties': {'mass': {'range': [0.5, 1.5], 'operation': 'scaling', 'distribution': 'uniform', 'schedule': 'linear', 'schedule_steps': 30000}}, 'rigid_shape_properties': {'friction': {'num_buckets': 250, 'range': [0.7, 1.3], 'operation': 'scaling', 'distribution': 'uniform', 'schedule': 'linear', 'schedule_steps': 30000}}}}}}, 'sim': {'substeps': 2, 'physx': {'num_threads': 4, 'solver_type': 1, 'num_position_iterations': 8, 'num_velocity_iterations': 0, 'contact_offset': 0.002, 'rest_offset': 0.0, 'bounce_threshold_velocity': 0.2, 'max_depenetration_velocity': 1000.0, 'default_buffer_size_multiplier': 5.0}, 'flex': {'num_outer_iterations': 5, 'num_inner_iterations': 20, 'warm_start': 0.8, 'relaxation': 0.75}}, 'name': 'ShadowHandGrasp', 'headless': True}
# args Namespace(algo='ppo', backbone_type='', cfg_env='cfg/shadow_hand_grasp.yaml', cfg_train='cfg/ppo/config.yaml', checkpoint='Base', compute_device_id=0, datatype='random', device='cuda', device_id=0, episode_length=0, experiment='Base', expert_model_dir='', flex=False, freeze_backbone=False, graphics_device_id=0, headless=True, horovod=False, logdir='logs/test', max_iterations=0, metadata=False, minibatch_size=-1, model_dir='example_model/state_based_model.pt', num_envs=0, num_threads=0, ocd_tag='', physics_engine=SimType.SIM_PHYSX, physx=False, pipeline='gpu', play=False, randomize=False, resume=0, rl_device='cuda:0', seed=0, sim_device='cuda:0', sim_device_type='cuda', slices=0, steps_num=-1, subscenes=0, task='ShadowHandGrasp', task_type='Python', test=False, torch_deterministic=False, train=True, use_gpu=True, use_gpu_pipeline=True, vision=False)

if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    # args.vision = True
    # args.backbone_type = "pn"
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
    
