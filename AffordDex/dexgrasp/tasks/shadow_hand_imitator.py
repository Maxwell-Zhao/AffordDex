from __future__ import annotations

import os
import random
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import utils.torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul

from dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rot6d_to_aa,
    rot6d_to_quat,
    quat_to_aa,
)
from torch import Tensor
from tqdm import tqdm
import pickle
from tasks.hand_base.base_task import BaseTask
from dataset.shadow import ShadowRH

from unittest import TextTestRunner
import xxlimited
from matplotlib.pyplot import axis
import numpy as np
import os
import os.path as osp
import random
from pyparsing import And
from utils.torch_jit_utils import *
from utils.data_info import plane2euler
import pdb
from dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass

def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)

class ShadowImitator(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, training=True):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]
        self.training = training

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]

        num_obs = 236 + 64
        self.num_obs_dict = {"full_state": num_obs}
        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal",
                           "robot0:thdistal"]
        self.hand_center = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips) 
        self.use_vel_obs = False
        self.fingertip_obs = True
        num_states = 0
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 24 
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.headless = self.cfg["headless"] = headless
        self.objs_assets = {}
        self._global_dexhand_indices = None  # Unique indices corresponding to all envs in flattened array
        self._global_manip_obj_indices = None  # Unique indices corresponding to all envs in flattened array

        ################################# Vision #################################
        self.table_dims = gymapi.Vec3(1, 1, 0.6)
        self.segmentation_id = {
            'hand': 2,
            'object': 3,
            'goal': 4,
            'table': 1,
        }
        self.num_state_obs = self.cfg['env']['numObservations']
        self.camera_depth_tensor_list = []
        self.camera_rgb_tensor_list = []
        self.camera_seg_tensor_list = []
        self.camera_vinv_mat_list = []
        self.camera_proj_mat_list = []
        self.camera_handles = []
        self.num_cameras = len(self.cfg['env']['vision']['camera']['eye'])
        self._cfg_camera_props()
        self._cfg_camera_pose()
        
        self.num_envs = self.cfg['env']['numEnvs']
        self.env_origin = torch.zeros((self.num_envs, 3), dtype=torch.float)

        self.x_n_bar = self.cfg['env']['vision']['bar']['x_n']
        self.x_p_bar = self.cfg['env']['vision']['bar']['x_p']
        self.y_n_bar = self.cfg['env']['vision']['bar']['y_n']
        self.y_p_bar = self.cfg['env']['vision']['bar']['y_p']
        self.z_n_bar = self.cfg['env']['vision']['bar']['z_n']
        self.z_p_bar = self.cfg['env']['vision']['bar']['z_p']
        self.depth_bar = self.cfg['env']['vision']['bar']['depth']
        self.num_pc_downsample = self.cfg['env']['vision']['pointclouds']['numDownsample']
        self.num_pc_presample = self.cfg['env']['vision']['pointclouds']['numPresample']
        self.num_each_pt = self.cfg['env']['vision']['pointclouds']['numEachPoint']
        self.num_pc_flatten = self.num_pc_downsample * self.num_each_pt
        self.cfg['env']['numObservations'] += self.num_pc_flatten
        self.cfg['env']['numObservations'] += 2*self.num_pc_downsample
        self.tighten_steps = self.cfg["env"]["tightenSteps"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]
        ########################################################################

        super().__init__(cfg=self.cfg, enable_camera_sensors=False)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs + self.num_object_dofs)
        self.dof_force_tensor = self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
        
        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)
        
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self._root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        
        self._base_state = self._root_state[:, 0, :]
        # self.dexhand_root_state = self.root_state_tensor[:, self.dexhand_handle, :]
        self.apply_forces = torch.zeros((self.num_envs, self.rigid_body_states.shape[1], 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.rigid_body_states.shape[1], 3), device=self.device, dtype=torch.float)

        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0
        
        ########################################################################
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.total_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.running_progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.failure_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.error_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.reward_dict = {}
        
        env_ptr = self.envs[0]
        
        if self.viewer:
            self.mano_joint_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
                for i in range(self.dexhand.n_bodies)
            ]
        
        self.id = -1
        self._refresh()

        self.joint_sphere_geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))
        self._manip_obj_handle = self.gym.find_actor_handle(env_ptr, "manip_obj")
        self._manip_obj_root_state = self._root_state[:, self._manip_obj_handle, :]
        self._pos_control = torch.zeros((self.num_envs, self.num_shadow_hand_dofs), dtype=torch.float, device=self.device)
  
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_dexhand(self, spacing):
        # ==================================load shadow hand_ asset======================================
        asset_root = "../assets"
        # shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        # table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_files = "textures/texture_stone_stone_texture_0.jpg"
        table_texture_files = osp.join(asset_root, table_texture_files)
        if not os.path.exists(table_texture_files):
            raise FileNotFoundError(f"Texture file not found: {table_texture_files}")
        # try:
        #     self.table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)
        #     print(f"Texture loaded successfully. Handle: {table_texture_handle}")
        # except gymapi.GymError as e:
        #     print(f"Failed to load texture: {e}")
        self.table_texture_handle = None
        # table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)
        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("zzhezh", shadow_hand_asset_file)
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self.shadow_hand_asset = shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)
        
        rigid_body_names = self.gym.get_asset_rigid_body_names(shadow_hand_asset)
        dof_names = self.gym.get_asset_dof_names(shadow_hand_asset)
        print('rigid_body_names',rigid_body_names)
        print('dof_names',dof_names)
        # rigid_body_names ['robot0:hand mount', 'robot0:palm', 'robot0:ffknuckle', 'robot0:ffproximal', 
        # 'robot0:ffmiddle', 'robot0:ffdistal', 'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 
        # 'robot0:mfdistal', 'robot0:rfknuckle', 'robot0:rfproximal', 'robot0:rfmiddle', 'robot0:rfdistal', 
        # 'robot0:lfmetacarpal', 'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 'robot0:lfdistal', 
        # 'robot0:thbase', 'robot0:thproximal', 'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
            
        # dof_names ['robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0', 'robot0:MFJ3', 'robot0:MFJ2', 
        #            'robot0:MFJ1', 'robot0:MFJ0', 'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0', 
        #            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0', 'robot0:THJ4', 
        #            'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0']            
        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies) # 24
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes) # 20
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs) # 22
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators) # 18
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons) # 4
        # import pdb; pdb.set_trace()
        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # set shadow_hand dof properties
        self.shadow_hand_dof_props = shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        # 定义手部刚体名称映射
        body_names = {
            'wrist': 'robot0:wrist',
            'palm': 'robot0:palm',
            'thumb': 'robot0:thdistal',
            'index': 'robot0:ffdistal',
            'middle': 'robot0:mfdistal',
            'ring': 'robot0:rfdistal',
            'little': 'robot0:lfdistal'
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state":
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)


        # self.dexhand_handles {'robot0:palm': 1, 'robot0:ffknuckle': 2, 'robot0:ffproximal': 3, 
        # 'robot0:ffmiddle': 4, 'robot0:ffdistal': 5, 'robot0:lfmetacarpal': 14, 'robot0:lfknuckle': 15, 
        # 'robot0:lfproximal': 16, 'robot0:lfmiddle': 17, 'robot0:lfdistal': 18, 'robot0:mfknuckle': 6, 
        # 'robot0:mfproximal': 7, 'robot0:mfmiddle': 8, 'robot0:mfdistal': 9, 'robot0:rfknuckle': 10, 
        # 'robot0:rfproximal': 11, 'robot0:rfmiddle': 12, 'robot0:rfdistal': 13, 'robot0:thbase': 19,
        # 'robot0:thproximal': 20, 'robot0:thhub': 21, 'robot0:thmiddle': 22, 'robot0:thdistal': 23}

    def _create_envs(self, spacing, num_per_row):
        self.lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # import table
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
        self.table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)
        self.table_height = 0.6
     
        
        self.init_data()
        self.goal_cond = self.cfg["env"]["goal_cond"] # False
        self.random_prior = self.cfg['env']['random_prior'] # True
        self.random_time = self.cfg["env"]["random_time"] # True
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        self._create_dexhand(spacing)

        # ===================================create table asset=====================================
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, gymapi.AssetOptions())
        # 机械手初始位姿
        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.8)  # gymapi.Vec3(0.1, 0.1, 0.65)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        # 对象初始位姿
        # object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch([self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        # goal_start_pose = gymapi.Transform()
        # goal_start_pose.p = object_start_pose.p + self.goal_displacement
        # goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        # goal_start_pose.p.z -= 0.0

        # ========================================================================
        # compute aggregate size
        self.shadow_hands = []   # 存储机械手实例
        self.envs = []  # 环境句柄列表
        self.object_init_state = []  # 对象初始状态
        self.goal_init_state = []  # 目标初始状态
        self.hand_start_states = []  # 机械手初始状态
        self.hand_indices = []  # 机械手索引
        self.fingertip_indices = []  # 指尖刚体索引
        self.object_indices = []  # 对象索引
        self.goal_object_indices = []  # 目标索引
        self.table_indices = []  # 桌子索引

        # RandomLoad
        self.num_obj_per_env = 2
        self.num_actors_per_env = 2 + self.num_obj_per_env * 2
        self.init_object_waiting_pose = []
        self.init_goal_waiting_pose = []
        self.all_object_indices = []
        self.all_goal_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.object_scale_buf = {}

        for i in range(self.num_envs):
            # create env instance
            num_per_row = int(np.sqrt(self.num_envs))
            env_ptr = self.gym.create_env(self.sim, self.lower, self.upper, num_per_row)
            
            # object
            current_asset, sum_rigid_body_count, sum_rigid_shape_count, obj_scale, obj_mass = self._create_obj_assets(i)
            # 最大刚体数 = 手部刚体数*1 + 对象刚体数*2 + 桌子刚体1
            max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * sum_rigid_body_count + 1  ##
            # 最大形状数 = 手部形状数*1 + 对象形状数*2 + 桌子形状1 
            max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * sum_rigid_shape_count + 1  ##
        
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            # -------------------- 添加Shadow Hand机械手 --------------------
            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            # shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "dexhand", i, -1, 0)
            shadow_hand_actor = self.gym.create_actor(env_ptr, self.shadow_hand_asset, shadow_hand_start_pose, "dexhand", i, -1, 0)
            # 记录机械手初始状态（位置+四元数+速度）
            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, self.shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # randomize colors and textures for rigid body
            hand_color = [147/255, 215/255, 160/255]
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            for n in self.agent_index[0]:
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(*hand_color))

            # create fingertip force-torque sensors
            if self.obs_type == "full_state":
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)


            # add object
            obj_transf = self.demo_data["obj_trajectory"][i][0]
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(obj_transf[0,3], obj_transf[1,3], obj_transf[2,3])
            # object_handle = obj_actor = self.gym.create_actor(env_ptr, current_asset, object_start_pose, "manip_obj", i, 0)
            obj_aa = rotmat_to_aa(obj_transf[:3,:3])
            obj_aa_angle = torch.norm(obj_aa)
            obj_aa_axis = obj_aa / (obj_aa_angle + 1e-6)
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)
            object_handle = obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj", i, 0)
            obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)
            scene_objs = self.demo_data["scene_objs"][i]
            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)

            MAX_SCENE_OBJS = 5 + (0 if not self.headless else 0)
            for so_id in range(MAX_SCENE_OBJS - len(scene_objs)):
                scene_asset = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, scene_asset_options)
                # ? collision filter bit is always 0b11111111, never collide with anything (except the ground)
                a = self.gym.create_actor(
                    env_ptr,
                    scene_asset,
                    gymapi.Transform(),
                    f"scene_obj_{so_id +  len(scene_objs)}",
                    self.num_envs + 1,
                    0b1,
                )
                c = [
                    gymapi.Vec3(1, 1, 0.5),
                    gymapi.Vec3(0.5, 1, 1),
                    gymapi.Vec3(1, 0, 1),
                    gymapi.Vec3(1, 1, 0),
                    gymapi.Vec3(0, 1, 1),
                    gymapi.Vec3(0, 0, 1),
                    gymapi.Vec3(0, 1, 0),
                    gymapi.Vec3(1, 0, 0),
                ][so_id + len(scene_objs)]
                self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)
      
            if not self.headless:
                for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                    # print("joint_name: ", joint_name)
                    # print("joint_vis_id: ", joint_vis_id)
                    joint_name = self.dexhand.to_hand(joint_name)[0]
                    joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                    a = self.gym.create_actor(env_ptr, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1)

        
            self.obj_handle = obj_actor
            
            # self.gym.set_actor_scale(env_ptr, self.obj_handle, obj_scale)
            obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_handle)
            obj_props[0].mass = min(0.5, obj_props[0].mass)  # * we only consider the mass less than 500g
            # ? caculate mass by density
            if obj_mass is not None:
                obj_props[0].mass = obj_mass
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_handle, obj_props)
            
            # 记录物体初始状态
            self.object_init_state.append([pose.p.x, pose.p.y, pose.p.z,
                                           pose.r.x, pose.r.y, pose.r.z,
                                           pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            # # 记录目标物体初始状态（基于偏移量）
            goal_start_pose = gymapi.Transform()
            goal_start_pose.p = pose.p + self.goal_displacement
            goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

            goal_start_pose.p.z -= 0.0
            
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])
            # object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(obj_index)
            # self.gym.set_actor_scale(env_ptr, object_handle, 1.0)

            # add goal object
            # goal_asset_dict[id][scale_id]
            # 创建目标物体（小球表示目标位置）
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, self.table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, self.table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))
            
            # if self.aggregate_mode > 0:
            #     self.gym.end_aggregate(env_ptr)
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        # self.object_pose_history = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        
        # Setup sim handles
        env_ptr = self.envs[0]
        self.dexhand_handle = self.gym.find_actor_handle(env_ptr, "dexhand")
        self.dexhand_handles = {k: self.gym.find_actor_rigid_body_handle(env_ptr, self.dexhand_handle, k) for k in self.dexhand.body_names}
        self.dexhand_cf_weights = {k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand.body_names}

        self._global_dexhand_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.device,
        ).view(self.num_envs, -1)

    def init_data(self):
        # tracking data
        self.demo_dataset_dict = {}
        # dataset_list = ["grabdemo", "oakink2"]
        self.max_episode_length = 1500
        self.dexhand = ShadowRH()
        mujoco2gym_transf = np.eye(4)
        # mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi/2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self.table_height])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.device, dtype=torch.float32)
    
        from dataset.grab_dataset import GrabDemoDexHand
        from dataset.oakink2_dataset import OakInk2DatasetDexHandRH
        
        oakink2_data = OakInk2DatasetDexHandRH(
            manipdata_type='oakink2',
            device=self.device,
            mujoco2gym_transf=self.mujoco2gym_transf,
            max_seq_len=self.max_episode_length,
            dexhand=self.dexhand,
            embodiment='shadow',
        )        
        
        self.dataIndices = []
        retargeting_path = 'data/retargeting/OakInk-v2/mano2shadow_rh/'
        retargeting_pkl = os.listdir(retargeting_path)
        for p in retargeting_pkl:
            p_hash = os.path.split(p)[-1].split("_")[5][:5]
            num = os.path.split(p)[-1].split("@")[1].split(".")[0]
            pkl = f"{p_hash}@{num}"
            self.dataIndices.append(pkl)

        # self.dataIndices = self.dataIndices[59:]
        # self.dataIndices = self.dataIndices[52:]
        # self.dataIndices = self.dataIndices[6:]
        # self.dataIndices = self.dataIndices[50:]
        # self.dataIndices = self.dataIndices[20:]
        # self.dataIndices = self.dataIndices[15:]
        # # 2129        
        # self.dataIndices = [x for x in self.dataIndices if x != '4e42c@4']
        # self.dataIndices = [x for x in self.dataIndices if x != '211a2@0']
        # self.dataIndices = [x for x in self.dataIndices if x != '2bae9@8']
        # self.dataIndices = [x for x in self.dataIndices if x != '2bae9@9']
        # self.dataIndices = [x for x in self.dataIndices if x != '7c801@0']
        # self.dataIndices = [x for x in self.dataIndices if x != '3b1e6@5']
        # self.dataIndices = [x for x in self.dataIndices if x != '86fb0@6']
        # self.dataIndices = [x for x in self.dataIndices if x != '38754@1']
        # self.dataIndices = [x for x in self.dataIndices if x != '2bae9@1']
        # self.dataIndices = [x for x in self.dataIndices if x != '3b1e6@3']
        
        print(len(self.dataIndices), "data indices loaded from OakInk2 dataset")

        def segment_data(k):
            idx = self.dataIndices[k % len(self.dataIndices)]
            # oakink2_data[idx]
            
            # return oakink2_data[idx]
            try:
                oakink2_data[idx]
                return oakink2_data[idx]
            except Exception:
                self.dataIndices = [x for x in self.dataIndices if x != idx]
                print(f"KeyError: {idx} not found in oakink2_data")
                return segment_data(k)


        self.demo_data = [segment_data(i) for i in tqdm(range(self.num_envs))]
        
        print(len(self.dataIndices), "data indices loaded from OakInk2 dataset")
        # import pdb; pdb.set_trace()
        self.demo_data = self.pack_data(self.demo_data)

    def _create_obj_assets(self, i):
        obj_id = self.demo_data["obj_id"][i]

        if obj_id in self.objs_assets:
            current_asset = self.objs_assets[obj_id]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.max_linear_velocity = 50
            asset_options.max_angular_velocity = 100
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
            current_asset = self.gym.load_asset(self.sim, *os.path.split(self.demo_data["obj_urdf_path"][i]), asset_options)

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = 2.0  # * We increase the friction coefficient to compensate for missing skin deformation friction in simulation. See the Appx for details.
                element.rolling_friction = 0.05
                element.torsion_friction = 0.05
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            self.objs_assets[obj_id] = current_asset

        # object_asset_options.disable_gravity = True 
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(current_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(current_asset)
        self.num_object_dofs = self.gym.get_asset_dof_count(current_asset)
        object_dof_props = self.gym.get_asset_dof_properties(current_asset)
        
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props['lower'][i])
            self.object_dof_upper_limits.append(object_dof_props['upper'][i])

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
        # * load assigned scale and mass for the object if available
        if obj_id in oakink2_obj_scale:
            scale = oakink2_obj_scale[obj_id]
        else:
            scale = 1.0

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass

    def pack_data(self, data):
        packed_data = {}
        # packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data).squeeze()

        for k in data[0].keys():
            if "alt" in k:
                continue
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    mano_joints.append(
                        torch.concat(
                            [
                                d[k][self.dexhand.to_hand(j_name)[0]]
                                for j_name in self.dexhand.body_names
                                if self.dexhand.to_hand(j_name)[0] != "wrist"
                            ],
                            dim=-1,
                        )
                    )
                packed_data[k] = fill_data(mano_joints)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif type(data[0][k]) == np.ndarray:
                print(k,'key')
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]

        def to_cuda(x):
            if type(x) == torch.Tensor:
                return x.to(self.device)
            elif type(x) == list:
                return [to_cuda(xx) for xx in x]
            elif type(x) == dict:
                return {k: to_cuda(v) for k, v in x.items()}
            else:
                return x

        packed_data = to_cuda(packed_data)

        return packed_data

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        
        self.joints_state = torch.stack([self.rigid_body_states[:, self.dexhand_handles[k],:] for k in self.dexhand.body_names], dim=1)
        
        # self.gym.refresh_net_contact_force_tensor(self.sim)

    def compute_reward(self):
        # 从演示数据中获取当前时间步（progress_buf）的目标状态：
        # 手腕位置和旋转（转换为四元数）
        # 手腕线速度和角速度
        # 关节位置和速度
        target_state = {}
        # 最大演示长度
        max_length = torch.clip(self.demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        cur_wrist_pos = self.demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]   
        target_state["wrist_pos"] = cur_wrist_pos
        
        cur_wrist_rot = self.demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

        target_state["wrist_vel"] = self.demo_data["wrist_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_ang_vel"] = self.demo_data["wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        cur_joints_pos = self.demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = self.demo_data["mano_joints_velocity"][
            torch.arange(self.num_envs), cur_idx
        ].reshape(self.num_envs, -1, 3)

        # 计算功率消耗​
        power = torch.abs(torch.multiply(self.dof_force, self._qd)).sum(dim=-1)
        target_state["power"] = power.detach()                                                                            
        wrist_power = torch.abs(torch.sum(self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]* self.root_state_tensor[self.hand_indices, 7:10],dim=-1,))  # ? linear force * linear velocity
        wrist_power += torch.abs(torch.sum(self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]* self.root_state_tensor[self.hand_indices, 10:],dim=-1,))  # ? torque * angular velocity
        target_state["wrist_power"] = wrist_power.detach()
        target_state["object_pos"] = self.object_pos.detach()

        # 计算收紧因子（训练时）​
        if self.training:
            last_step = self.gym.get_frame_count(self.sim)
            # linear_decay
            self.tighten_factor = 0.7
            scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
  
        else:
            scale_factor = 1.0

        last_step = self.gym.get_frame_count(self.sim)

        scale_factor = (np.e * 2) ** (-1 * last_step / 3000) * (1 - 0.7) + 0.7
        
        # target_state["joints_pos"] torch.Size([4096, 22, 3])
        rewards, resets, successes, _, _ = (
            compute_imitation_reward(
                self.reset_buf,
                self.progress_buf,
                self.running_progress_buf,
                self.root_state_tensor[self.hand_indices, :],
                self.joints_state,
                self.shadow_hand_dof_vel,
                target_state,
                max_length,
                scale_factor,
                self.dexhand.weight_idx,
            )
        )
        
        self.rew_buf[:] = rewards.detach()
        self.reset_buf[:] = resets.detach()
        self.successes[:] = successes.detach()
        # print('reward_dict',reward_dict)
        # pdb.set_trace()
        self.extras['successes'] = self.successes.detach()
        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        # 物体状态
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        
        # 手掌状态
        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # 手指状态
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                              
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        idx = self.hand_body_idx_dict['little']
        self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # 指尖状态
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # 世界坐标系与物体坐标系之间的转换
        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        # 旋转转换
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)
        
        # 手掌相对于目标物体的位置偏差
        self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        # 手掌相对于物体的旋转
        self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        # 旋转偏差（相对姿态）
        self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        # 关节位置偏差
        self.delta_qpos = self.shadow_hand_dof_pos - self.target_qpos

        self.compute_full_state()

    def compute_full_state(self):
        self.get_unpose_quat()
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##
        # 0:66
        # 机械臂的数据
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        # 关节速度
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        # 关节力矩
        self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]
        # 指尖状态
        fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(5):
            aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 66:131: ft states
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

        # 131:161: ft sensors: do not need repose
        # 指尖力传感器
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        hand_pose_start = fingertip_obs_start + 95
        # 161:167: hand_pose
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
        euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)

        # 动作历史观测
        action_obs_start = hand_pose_start + 6
        # 167:191: action
        aux = self.actions[:, :24]
        aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        self.obs_buf[:, action_obs_start:action_obs_start + 24] = aux

        # 物体状态观测
        obj_obs_start = action_obs_start + 24  # 144
        # 191:207 object_pose, goal_pos
        # if (self.object_pose_history == None):
        #     self.object_pose_history = self.object_pose
               
            
        self.obs_buf[:, obj_obs_start : obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)

        # self.obs_buf[:, obj_obs_start+16:obj_obs_start + 19] = self.unpose_point(self.object_pose_history[:, 0:3])
        # self.obs_buf[:, obj_obs_start + 19:obj_obs_start + 23] = self.unpose_quat(self.object_pose_history[:, 3:7])
        # self.object_pose_history = self.object_pose.detach().clone()
        # 207:236 goal
        # In UniDexGrasp++, we don't use the target goal grasp pose so we simply set
        # this observation all to zero
        hand_goal_start = obj_obs_start + 16
        self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = 0 # self.delta_target_hand_pos
        self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = 0 # self.delta_target_hand_rot
        self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = 0 # self.delta_qpos

        # 236: visual feature
        # 视觉特征
        visual_feat_start = hand_goal_start + 29
        # 236: 300: visual feature
        # self.obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf
        self.obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0
        
        return

    def pre_physics_step(self, actions):
        # ? >>> for visualization
        if self.viewer:
            cur_idx = self.progress_buf
            self.gym.clear_lines(self.viewer)
            # self.draw_coordinate_frame(self.viewer, self.envs[0], self.gym)
            cur_wrist_pos = self.demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
            cur_mano_joint_pos = self.demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx].reshape(self.num_envs, -1, 3)
            cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = cur_mano_joint_pos[:, k]
                # self.draw_rigid_bodies_demo()
            for env_id, env_ptr in enumerate(self.envs):
                # for k in self.dexhand.body_names:
                #     self.set_force_vis(env_ptr, k, torch.norm(self.net_cf[env_id, self.dexhand_handles[k]], dim=-1) != 0)

                def add_lines(viewer, env_ptr, hand_joints, color):
                    assert hand_joints.shape[0] == self.dexhand.n_bodies and hand_joints.shape[1] == 3
                    hand_joints = hand_joints.cpu().numpy()
                    lines = np.array([[hand_joints[b[0]], hand_joints[b[1]]] for b in self.dexhand.bone_links])
                    for line in lines:
                        self.gym.add_lines(viewer, env_ptr, 1, line, color)

                color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32) * 2.0
                add_lines(self.viewer, env_ptr, cur_mano_joint_pos[env_id].cpu(), color)
    
        ###############################################################           
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        # if only goals need reset, then call set API
        # if len(goal_env_ids) > 0 and len(env_ids) == 0:
        #     self.reset_target_pose(goal_env_ids, apply_reset=True)
        # # if goals need reset in addition to other envs, call set API in reset()
        # elif len(goal_env_ids) > 0:
        #     self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids)
        
        self.get_pose_quat()
        actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        self.actions = actions.detach().clone().to(self.device)

        self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
        self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])


        self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
        self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
                                                gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def pose_vec(self, vec):
        return quat_apply(self.pose_z_theta_quat, vec)
    
    def get_pose_quat(self):
        self.pose_z_theta_quat = quat_from_euler_xyz(
            torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
            self.z_theta,
        )
        return

    def reset(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        # self.reset_target_pose(env_ids)
        last_step = self.gym.get_frame_count(self.sim)

        # if self.training and last_step >= self.tighten_steps:
        # # if last_step >= self.tighten_steps:
        #     running_steps = self.running_progress_buf[env_ids] - 1
        #     max_running_steps, max_running_idx = running_steps.max(dim=0)
        #     max_running_env_id = env_ids[max_running_idx]
            # if max_running_steps > self.best_rollout_len:
            #     self.best_rollout_len = max_running_steps
            #     self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps
        if self.random_state_init:
            seq_idx = torch.floor(
                self.demo_data["seq_len"][env_ids]
                * 0.98
                * torch.rand_like(self.demo_data["seq_len"][env_ids].float())
            ).long()
            seq_idx = torch.clamp(seq_idx, torch.zeros(1 ,device=self.device).long(), torch.floor(self.demo_data["seq_len"][env_ids] * 0.98).long())
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data["seq_len"][env_ids].long())
            else:
                seq_idx = torch.zeros_like(self.demo_data["seq_len"][env_ids].long())

        dof_pos = self.demo_data["opt_dof_pos"][env_ids, seq_idx]
        dof_vel = self.demo_data["opt_dof_velocity"][env_ids, seq_idx]

        opt_wrist_pos = self.demo_data["opt_wrist_pos"][env_ids, seq_idx]
        opt_wrist_rot = aa_to_quat(self.demo_data["opt_wrist_rot"][env_ids, seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = self.demo_data["opt_wrist_velocity"][env_ids, seq_idx]
        opt_wrist_ang_vel = self.demo_data["opt_wrist_angular_velocity"][env_ids, seq_idx]

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)
        self._q[env_ids, :] = dof_pos
        self._qd[env_ids, :] = dof_vel
        self._base_state[env_ids, :] = opt_hand_pose_vel
        self._pos_control[env_ids, :] = dof_pos

        # import pdb; pdb.set_trace()
        
        # dof_pos torch.Size([2, 22])
        # dof_vel torch.Size([2, 22])
        # opt_wrist_pos torch.Size([2, 3])
        # opt_wrist_rot torch.Size([2, 4])
        # opt_wrist_vel torch.Size([2, 3])
        # opt_wrist_ang_vel torch.Size([2, 3])
        # opt_hand_pose_vel torch.Size([2, 13])
        # self._q torch.Size([2, 22])
        # self._qd torch.Size([2, 22])
        # self._base_state torch.Size([2, 13])
        # reset manip obj
        obj_pos_init = self.demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
        obj_rot_init = self.demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
        # [w, x, y, z] to [x, y, z, w]
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        obj_vel = self.demo_data["obj_velocity"][env_ids, seq_idx]
        obj_ang_vel = self.demo_data["obj_angular_velocity"][env_ids, seq_idx]

        self._manip_obj_root_state[env_ids, :3] = obj_pos_init
        self._manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        self._manip_obj_root_state[env_ids, 7:10] = obj_vel
        self._manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

        dexhand_multi_env_ids_int32 = self._global_dexhand_indices[env_ids].flatten()
        manip_obj_multi_env_ids_int32 = self._global_manip_obj_indices[env_ids].flatten()
        
        if torch.isnan(self._dof_state).any() or torch.isinf(self._dof_state).any():
            print("!!!!!!!!!!!!!!!!!!!!!!!!! NaN/Inf in _dof_state !!!!!!!!!!!!!!!!!!!!!!!!!")
            import pdb; pdb.set_trace()

        
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        
        # reset object
        # self.root_state_tensor[self.object_indices[env_ids]] = self._manip_obj_root_state[env_ids].clone()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32), len(dexhand_multi_env_ids_int32)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(manip_obj_multi_env_ids_int32), len(manip_obj_multi_env_ids_int32)
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            # gymtorch.unwrap_tensor(self.prev_targets),
             gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        # print('self.progress_buf[env_ids] = seq_idx',seq_idx)
        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        # self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0

        # self.tips_contact_history[env_ids] = torch.ones_like(self.tips_contact_history[env_ids]).bool()

    def post_physics_step(self):
        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1
        
        self.compute_observations()
        self.compute_reward()

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

    def _cfg_camera_props(self):
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.enable_tensors = True
        return

    def _cfg_camera_pose(self):
        self.camera_eye_list = []
        self.camera_lookat_list = []
        camera_eye_list = self.cfg['env']['vision']['camera']['eye']
        camera_lookat_list = self.cfg['env']['vision']['camera']['lookat']
        table_centor = np.array([0.0, 0.0, self.table_dims.z])
        for i in range(self.num_cameras):
            camera_eye = np.array(camera_eye_list[i]) + table_centor
            camera_lookat = np.array(camera_lookat_list[i]) + table_centor
            self.camera_eye_list.append(gymapi.Vec3(*list(camera_eye)))
            self.camera_lookat_list.append(gymapi.Vec3(*list(camera_lookat)))
        return
    
    def get_unpose_quat(self):
        self.unpose_z_theta_quat = quat_from_euler_xyz(
            torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
            -self.z_theta,
        )
        return

    def unpose_point(self, point):
        return self.unpose_vec(point)

    def unpose_vec(self, vec):
        return quat_apply(self.unpose_z_theta_quat, vec)
    
    def unpose_quat(self, quat):
        return quat_mul(self.unpose_z_theta_quat, quat)
   
    def unpose_state(self, state):
        state = state.clone()
        state[:, 0:3] = self.unpose_point(state[:, 0:3])
        state[:, 3:7] = self.unpose_quat(state[:, 3:7])
        state[:, 7:10] = self.unpose_vec(state[:, 7:10])
        state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

    def draw_rigid_bodies_demo(self, env_id=1):
        joint_positions = []
        for joint_tensor in self.mano_joint_points:
            pos = joint_tensor[env_id,:3].detach().cpu().numpy()
            # pos = pos + np.array([0, 0, self.table_height])
            joint_positions.append(pos)
        # joint_positions = np.array(joint_positions)
        # joint_positions = torch.tensor(self.mano_joint_points)[:,:,0:3]
        color = np.array([1, 0, 0])
        radius = 0.02
        for pos in joint_positions:
            vertices = np.array([
                [pos[0], pos[1], pos[2]],
                [pos[0] + radius, pos[1], pos[2]],
                [pos[0], pos[1] + radius, pos[2]],
                [pos[0], pos[1], pos[2] + radius]
            ], dtype=np.float32)
            colors = np.array([[1,0,0] for _ in range(len(vertices))], dtype=np.float32)

            self.gym.add_lines(self.viewer, self.envs[env_id], len(vertices), vertices, colors)

    def draw_coordinate_frame(self,viewer, env, gym, length=10):
        vertices = [
            [0, 0, 0], [length, 0, 0],  
            [0, 0, 0], [0, length, 0],
            [0, 0, 0], [0, 0, length] 
        ]
        
        colors = [
            [1, 0, 0], [1, 0, 0],  # 红色X轴
            [0, 1, 0], [0, 1, 0],  # 绿色Y轴
            [0, 0, 1], [0, 0, 1]   # 蓝色Z轴
        ]

        gym.add_lines(viewer, env, len(vertices)//2, vertices, colors)

    def step(self, actions, id):
        self.id = id
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self.pre_physics_step(actions)
        
        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
 
        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
            self.obs_buf2 = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf2)

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    root_state_tensor,
    joints_state,
    shadow_hand_dof_vel,
    target_states: Dict[str, Tensor],
    max_length, 
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]]):
    
    # end effector pose reward
    current_eef_pos = root_state_tensor[:, :3]
    current_eef_quat = root_state_tensor[:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = root_state_tensor[:, 7:10]
    current_eef_ang_vel = root_state_tensor[:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = joints_state[:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints    
    # diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    # diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    # diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    # diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    # diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = joints_state[:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_vel = shadow_hand_dof_vel

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
    )  # sanity check

    failed_execute = (
        (
            # (diff_thumb_tip_pos_dist > 0.15 / 0.7 * scale_factor)
            # | (diff_index_tip_pos_dist > 0.15 / 0.7 * scale_factor)
            # | (diff_middle_tip_pos_dist > 0.15 / 0.7 * scale_factor)
            # | (diff_pinky_tip_pos_dist > 0.15 / 0.7 * scale_factor)
            # | (diff_ring_tip_pos_dist > 0.15 / 0.7 * scale_factor)
            (diff_level_1_pos_dist > 0.15 / 0.7 * scale_factor)
            | (diff_level_2_pos_dist > 0.15 / 0.7 * scale_factor)
        )
        & (running_progress_buf >= 50)
    ) | error_buf
    reward_execute = (
        0.1 * reward_eef_pos
        + 0.6 * reward_eef_rot
        + 0.6 * reward_level_1_pos
        + 0.8 * reward_level_2_pos
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.05 * reward_power
        + 0.05 * reward_wrist_power
    )
    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
    
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_level_1_pos": reward_level_1_pos,
        "reward_level_2_pos": reward_level_2_pos,
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
    }
    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot