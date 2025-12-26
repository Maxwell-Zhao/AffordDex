import os
import pickle
from functools import lru_cache

import numpy as np
import smplx
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from torch.utils.data import Dataset

from dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa, quat_to_aa
from scipy.ndimage import gaussian_filter1d
from termcolor import cprint
from manotorch.manolayer import ManoLayer, MANOOutput
import trimesh
from dataset.base import ManipData

def dump_obj_mesh(filename, vertices, faces=None):
    assert vertices.shape[1] == 3 and (faces is None or faces.shape[1] == 3)
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if faces is not None:
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy().astype(np.int32)
        else:
            faces = faces.astype(np.int32)
    with open(filename, "w") as obj_file:
        for v in vertices:
            obj_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        if faces is not None:
            for f in faces + 1:
                obj_file.write("f {} {} {}\n".format(f[0], f[1], f[2]))

class GrabDemoDexHand(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/grab_demo/102",
        split: str = "all",
        skip: int = 1,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        self.manolayer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=None,
            mano_assets_root="data/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

        self.data_pathes = [os.path.join(self.data_dir, "102_sv_dict.npy")]

        self.device = device

        transf_offset = np.eye(4)
        transf_offset[:3, :3] = aa_to_rotmat(np.array([-np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        transf_offset[:3, 3] = np.array([0.0, 0.018, 0.0])
        # print('transf_offset',transf_offset.mean())

        self.transf_offset = torch.tensor(transf_offset, dtype=torch.float32, device=mujoco2gym_transf.device)
        # print('mujoco2gym_transf.device',mujoco2gym_transf.device)
        # print('self.mujoco2gym_transf',self.mujoco2gym_transf.mean())
        
        self.mujoco2gym_transf = mujoco2gym_transf @ self.transf_offset
        # print('mujoco2gym_transf',mujoco2gym_transf.mean())
        # print('self.mujoco2gym_transf',self.mujoco2gym_transf.mean())
        # print('2'*30)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data_pathes)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        assert (
            idx == "g0"
        ), "We directly borrow the grab demo data from https://github.com/Meowuu7/QuasiSim, so we only support idx='g0'"
        idx = int(idx[1:])
        assert self.mujoco2gym_transf is not None

        data = np.load(self.data_pathes[idx], allow_pickle=True).item()
        obj_mesh = trimesh.load(os.path.join(self.data_dir, "102_obj.obj"), process=False)

        length = len(data["object_global_orient"])
        obj_pose = np.eye(4)[None].repeat(length, axis=0)
        obj_pose[:, :3, :3] = aa_to_rotmat(data["object_global_orient"]).transpose(0, 2, 1)  # ?
        obj_pose[:, :3, 3] = data["object_transl"]
        obj_pose = torch.tensor(obj_pose, device=self.device)
        hand_rot = torch.tensor(data["rhand_global_orient_gt"], device=self.device)
        hand_tsl = torch.tensor(data["rhand_transl"], device=self.device)
        mano_out_verts = torch.tensor(data["rhand_verts"], device=self.device)
        mano_out_joints = torch.matmul(self.manolayer.th_J_regressor, mano_out_verts)

        wrist_pos = mano_out_joints.detach()[:, 0]
        middle_pos = mano_out_joints.detach()[:, 4]

        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # TODO hack for wrist position

        mano_joints = {
            "index_proximal": mano_out_joints.detach()[:, 1],
            "index_intermediate": mano_out_joints.detach()[:, 2],
            "index_distal": mano_out_joints.detach()[:, 3],
            "index_tip": mano_out_verts[:, 353].detach(),  # reselect tip
            "middle_proximal": mano_out_joints.detach()[:, 4],
            "middle_intermediate": mano_out_joints.detach()[:, 5],
            "middle_distal": mano_out_joints.detach()[:, 6],
            "middle_tip": mano_out_verts[:, 467].detach(),  # reselect tip
            "pinky_proximal": mano_out_joints.detach()[:, 7],
            "pinky_intermediate": mano_out_joints.detach()[:, 8],
            "pinky_distal": mano_out_joints.detach()[:, 9],
            "pinky_tip": mano_out_verts[:, 695].detach(),  # reselect tip
            "ring_proximal": mano_out_joints.detach()[:, 10],
            "ring_intermediate": mano_out_joints.detach()[:, 11],
            "ring_distal": mano_out_joints.detach()[:, 12],
            "ring_tip": mano_out_verts[:, 576].detach(),  # reselect tip
            "thumb_proximal": mano_out_joints.detach()[:, 13],
            "thumb_intermediate": mano_out_joints.detach()[:, 14],
            "thumb_distal": mano_out_joints.detach()[:, 15],
            "thumb_tip": mano_out_verts[:, 766].detach(),  # reselect tip
        }

        inspire_rot_offset = self.dexhand.relative_rotation
        wrist_rot = aa_to_rotmat(hand_rot) @ torch.tensor(
            np.repeat(inspire_rot_offset[None], length, axis=0), device=self.device
        )

        mesh = Meshes(
            verts=torch.from_numpy(obj_mesh.vertices[None, ...]).float(),
            faces=torch.from_numpy(obj_mesh.faces[None, ...]).float(),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": self.data_pathes[idx],
            "obj_id": "-1",  # ? placeholder
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": os.path.join(self.data_dir, "102_obj.urdf"),
            "obj_trajectory": torch.tensor(
                np.stack(obj_pose[:: self.skip].cpu()), device=self.device, dtype=torch.float
            ),
            "scene_objs": [],  # ? placeholder
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        self.process_data(data, idx, rs_verts_obj)
        opt_path = f"data/retargeting/grab_demo/mano2{str(self.dexhand)}/102_sv_dict.pkl"

        self.load_retargeted_data(data, opt_path)

        return data


if __name__ == "__main__":
    fdata = GrabDemoDexHand(mujoco2gym_transf=torch.eye(4))
    print(fdata[0])