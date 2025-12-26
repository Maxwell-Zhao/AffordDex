import multiprocessing
import trimesh
import numpy as np
import sys
import os
import argparse
import coacd
from termcolor import cprint
import glob

def coacd_process(
    input,
    output,
    quiet=False,
    threshold=0.05,
    preprocess_mode="auto",
    resolution=2000,
    no_merge=False,
    decimate=False,
    max_ch_vertex=256,
    extrude=False,
    extrude_margin=0.01,
    max_convex_hull=-1,
    mcts_iteration=150,
    mcts_max_depth=3,
    mcts_node=20,
    prep_resolution=50,
    pca=False,
    apx_mode="ch",
    seed=0,
):
    if not os.path.isfile(input):
        print(input, "is not a file")
        exit(1)

    if quiet:
        coacd.set_log_level("error")

    try:
        mesh = trimesh.load(input, force="mesh", process=False, skip_materials=True)
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        result = coacd.run_coacd(
            mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode=preprocess_mode,
            preprocess_resolution=prep_resolution,
            resolution=resolution,
            mcts_nodes=mcts_node,
            mcts_iterations=mcts_iteration,
            mcts_max_depth=mcts_max_depth,
            pca=pca,
            merge=not no_merge,
            decimate=decimate,
            max_ch_vertex=max_ch_vertex,
            extrude=extrude,
            extrude_margin=extrude_margin,
            apx_mode=apx_mode,
            seed=seed,
        )
        mesh_parts = []
        for vs, fs in result:
            mesh_parts.append(trimesh.Trimesh(vs, fs))
        scene = trimesh.Scene()
        np.random.seed(0)
        for p in mesh_parts:
            # p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
            scene.add_geometry(p)
        
        print(f"Exporting to {output}", "green")
        scene.export(output)
        open(output + ".done", "w").close() # touch done file
    except:
        print(f"Failed to process {input}", "red")
        open(output + ".failed", "w").close()           

def get_folder_names_recursive(root_dir):
    folder_names = set() 
    
    def _scan(current_dir):
        with os.scandir(current_dir) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    folder_names.add(entry.name)
                    _scan(entry.path)
    
    _scan(root_dir)
    return sorted(folder_names)  

def process_coacd(args):
    input_file = args.input
    output_file = args.output

    if not os.path.isfile(input_file):
        print(input_file, "is not a file")
        exit(1)

    if args.quiet:
        coacd.set_log_level("error")

    mesh = trimesh.load(input_file, force="mesh", process=False, skip_materials=True)
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        preprocess_mode=args.preprocess_mode,
        preprocess_resolution=args.prep_resolution,
        resolution=args.resolution,
        mcts_nodes=args.mcts_node,
        mcts_iterations=args.mcts_iteration,
        mcts_max_depth=args.mcts_max_depth,
        pca=args.pca,
        merge=not args.no_merge,
        decimate=args.decimate,
        max_ch_vertex=args.max_ch_vertex,
        extrude=args.extrude,
        extrude_margin=args.extrude_margin,
        apx_mode=args.apx_mode,
        seed=args.seed,
    )
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        # p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)  # 使用exist_ok防止已存在报错
    scene.export(output_file)
    print("Exported to", output_file)
    open(output_file + ".done", "w").close() # touch done file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",type=str,default='/data/zhaohaoyu/UniDexGrasp2/OakInk-v2/datasets--kelvin34501--OakInk-v2/object_preview/align_ds/O02@0015@00014/scan.ply',help="input model loaded by trimesh. Supported formats: glb, gltf, obj, off, ply, stl, etc.",)
    parser.add_argument("-o","--output",type=str,default='/data/zhaohaoyu/UniDexGrasp2/OakInk-v2/datasets--kelvin34501--OakInk-v2/coacd_object_preview/align_ds/O02@0015@00014/coacd.ply',help="output model exported by trimesh. Supported formats: glb, gltf, obj, off, ply, stl, etc.",)
    parser.add_argument("--quiet", default=True, action="store_true", help="do not print logs")
    parser.add_argument("-t","--threshold",type=float,default=0.07,help="termination criteria in [0.01, 1] (0.01: most fine-grained; 1: most coarse)",)
    parser.add_argument("-pm","--preprocess-mode",type=str,default="auto",help="No remeshing before running CoACD. Only suitable for manifold input.",)
    parser.add_argument("-r","--resolution",type=int,default=2000,help="surface samping resolution for Hausdorff distance computation",)
    parser.add_argument("-nm","--no-merge",action="store_true",help="If merge is enabled, try to reduce total number of parts by merging.",)
    parser.add_argument("-d","--decimate",action="store_true",help="If decimate is enabled, reduce total number of vertices per convex hull to max_ch_vertex.",)
    parser.add_argument("-dt","--max-ch-vertex",type=int,default=32,help="max # vertices per convex hull, works only when decimate is enabled",)
    parser.add_argument("-ex","--extrude",action="store_true",help="If extrude is enabled, extrude the neighboring convex hulls along the overlap face (other faces are unchanged).",)
    parser.add_argument("-em","--extrude-margin",type=float,default=0.01,help="extrude margin, works only when extrude is enabled",)
    parser.add_argument("-c","--max-convex-hull",type=int,default=-1,help="max # convex hulls in the result, -1 for no limit, works only when merge is enabled",)
    parser.add_argument("-mi","--mcts-iteration",type=int,default=2000,help="Number of MCTS iterations.",)
    parser.add_argument("-md","--mcts-max-depth",type=int,default=5,help="Maximum depth for MCTS search.",)
    parser.add_argument("-mn","--mcts-node",type=int,default=20,help="Number of cut candidates for MCTS.",)
    parser.add_argument("-pr","--prep-resolution",type=int,default=50,help="Preprocessing resolution.",)
    parser.add_argument("--pca",action="store_true",help="Use PCA to align input mesh. Suitable for non-axis-aligned mesh.",)
    parser.add_argument("-am","--apx-mode",type=str,default="ch",help="Approximation shape mode (ch/box).",)
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count() // 2,help="并行任务数 (默认使用CPU数的一半)")

    args = parser.parse_args()
    
    root_dir = '/data/zhaohaoyu/UniDexGrasp2/OakInk-v2/datasets--kelvin34501--OakInk-v2/object_preview/align_ds/'
    res_dir = '/data/zhaohaoyu/UniDexGrasp2/OakInk-v2/datasets--kelvin34501--OakInk-v2/coacd_object_preview/align_ds/'

    folder_names = get_folder_names_recursive(root_dir)

    for name in folder_names:
        path = os.path.join(root_dir, name)
        ply_name = glob.glob(os.path.join(path, '**', '*.ply'), recursive=True)
        obj_name = glob.glob(os.path.join(path, '**', '*.obj'), recursive=True)
        if len(ply_name)!=0:
            args.input = os.path.join(path, ply_name[0])
            args.output = os.path.join(res_dir, name, ply_name[0])
            process_coacd(args)
        elif len(obj_name)!=0:
            args.input = os.path.join(path, obj_name[0])
            args.output = os.path.join(res_dir, name, ply_name[0])
            process_coacd(args)
        else:
            print(f"Warning: No .ply or .obj file found in {path}")
            continue