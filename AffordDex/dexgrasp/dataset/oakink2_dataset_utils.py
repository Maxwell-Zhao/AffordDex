import trimesh
import os
from typing import List


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) != 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values())
        )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_obj_map(obj_prefix: str, obj_list: List[str]):
    res = {}
    for obj_id in obj_list:
        obj_filedir = os.path.join(obj_prefix, obj_id)
        candidate_list = [el for el in os.listdir(obj_filedir) if os.path.splitext(el)[-1] in [".obj", ".ply"]]
        assert len(candidate_list) == 1
        obj_filename = candidate_list[0]
        obj_filepath = os.path.join(obj_filedir, obj_filename)
        if os.path.splitext(obj_filename)[-1] == ".obj":
            mesh = as_mesh(trimesh.load_mesh(obj_filepath, process=False, skip_materials=True, force="mesh"))
        else:
            mesh = trimesh.load(obj_filepath, process=False, force="mesh")
        res[obj_id] = (mesh, obj_filepath)
    return res


oakink2_obj_scale = {
    "O02@0206@00001": 1.15,
    "O02@0206@00002": 0.95,
    "O02@0029@00011": 0.9,
    "O02@0029@00012": 1.2,
    "O02@0015@00020": 0.98,
    "O02@0015@00019": 1.02,
}
oakink2_obj_mass = {
    "O02@0015@00002": 0.101,
    "O02@0015@00001": 0.027,
    "C12001": 0.114,
    "O02@0030@00002": 0.0144,
    "O02@0033@00002": 0.014,
    "O02@0011@00003": 0.12,
    "O02@0206@00002": 0.163,
}  # TODO: add more objects
