# pointcloud_segmentation_clip_fgvp.py

import os
import json
import re
import base64
import torch
import openai
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import clip
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from point_utils import project_pcd, get_depth_map, mask_pcd_2d
from fine_grained_visual_prompt import FGVP_ENSEMBLE
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



def filter_masks_and_boxes(masks, boxes, centers, max_area_ratio=0.4):
    N, _, H, W = masks.shape
    total_area = H * W
    keep = []
    for i, m in enumerate(masks):
        if (m > 0).sum().item() / total_area < max_area_ratio:
            keep.append(i)
    if len(keep) == 0:
        return masks, boxes, centers  # 不过滤任何，避免报错
    
    return masks[keep], boxes[keep], centers[keep]

openai.api_key = "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"  # 请替换成你自己的 OpenAI API Key

GPT_PROMPT = """You are an expert in identifying which part of a physical object should not be touched, especially in robotic grasping tasks. 

You will be given 6-view images of an object. Your task is:
- Identify one non-touchable part.
- from image name get what object it is.

Return one sentence only, following this format:
"This is a _____. _____ of ______ should not be touched."
"""

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def query_gpt_description(image_dict, object_name):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Images of an object named '{object_name}'.\n\n" + GPT_PROMPT},
            *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in image_dict.values()]
        ]
    }]
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=1024
    )
    return response['choices'][0]['message']['content']

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def query_gpt_description(image_dict, object_name):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Images of an object named '{object_name}'.\n\n" + GPT_PROMPT},
            *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in image_dict.values()]
        ]
    }]
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=512
    )
    return response['choices'][0]['message']['content']

import re

def extract_non_touchable_text(desc):
    match = re.search(r"([A-Za-z\s\-]+)\s+should not be touched", desc.lower())
    if match:
        phrase = match.group(1).strip()
        # 去掉开头的 "the "，如果有的话
        if phrase.startswith("the "):
            phrase = phrase[4:]
        return phrase
    return "target region"


class ClipModel(torch.nn.Module):
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def forward(self, image, text, softmax=False):
        tokenized_text = self.tokenizer(text).to(self.device)
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(tokenized_text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        if softmax:
            similarity = (100 * similarity).softmax(-1)
        return similarity

class PointCloudSegmentationTask:
    def __init__(self, name, image_root, pcd_root, output_root, sam_ckpt):
        self.name = name
        self.image_root = image_root
        self.pcd_path = os.path.join(pcd_root, f"{name.removesuffix('_uv')}.ply")
        self.cam_path = os.path.join(image_root, f"{name}_camera_params.json")
        self.output_root = os.path.join(output_root, name)
        os.makedirs(self.output_root, exist_ok=True)
        self.image_paths = {
            v: os.path.join(image_root, f"{name}_{v}.png")
            for v in ["front", "back", "left", "right", "top", "bottom"]
        }
        self._load()
        self._init_models(sam_ckpt)

    def _load(self):
        self.pcd = o3d.t.io.read_point_cloud(self.pcd_path)
        legacy = self.pcd.to_legacy()
        self.points = np.asarray(legacy.points)
        bbox = legacy.get_axis_aligned_bounding_box()
        self.obj_size = np.max(bbox.get_max_bound() - bbox.get_min_bound())
        with open(self.cam_path) as f:
            self.cam_params = json.load(f)

    def _init_models(self, sam_ckpt):
        self.device = torch.device("cuda")
        self.sam_model = sam_model_registry['vit_h'](checkpoint=sam_ckpt).to(self.device)
        self.sam_generator = SamAutomaticMaskGenerator(
            self.sam_model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.7,
            stability_score_offset=0.7,
            box_nms_thresh=0.7,
            crop_n_layers=0,
            crop_nms_thresh=0.7,
            crop_overlap_ratio=512 / 1500,         # ≈ 0.341
            crop_n_points_downscale_factor=1,
            point_grids=None,
            min_mask_region_area=0,
            output_mode="binary_mask",
        )

        encoder, _ = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_model = ClipModel(encoder, clip.tokenize, self.device)

        self.fgvp = FGVP_ENSEMBLE(
            color_line='red', thickness=2, color_mask='green', alpha=0.5,
            clip_processing='resize', clip_image_size=336,
            resize_transform_clip=ResizeLongestSide(336),
            pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1).to(self.device) * 255,
            pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1).to(self.device) * 255,
            blur_std_dev=100, mask_threshold=self.sam_model.mask_threshold,
            contour_scale=1.0, device=self.device)

    def run(self):
        image_dict = {k: encode_image_to_base64(v) for k, v in self.image_paths.items() if os.path.exists(v)}
        #desc = query_gpt_description(image_dict, self.name)
        desc = "This is a knife. The sharp, steel blade of the knife should not be touched."
        text_prompt = "sharp, steel blade of the knife"
        print(f"[GPT] {text_prompt}")

        self.masks = []
        for view, path in self.image_paths.items():
            if not os.path.exists(path): continue
            image = cv2.imread(path)
            
            outputs = self.sam_generator.generate(image)
            masks = torch.from_numpy(np.stack([x['segmentation'] for x in outputs])).unsqueeze(1).to(self.device)
            boxes = torch.tensor(np.stack([x['bbox'] for x in outputs])).float().to(self.device)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
            masks, boxes, centers = filter_masks_and_boxes(masks, boxes, centers, max_area_ratio=0.045)
            clip_inputs = torch.cat([
                self.fgvp("blur_mask", image[:, :, ::-1], centers, boxes, masks)
            ])
            logits = self.clip_model(clip_inputs, [f"a photo of a {text_prompt}"])
            best = logits.argmax().item()
            self.masks.append((masks[best].squeeze(), view))

            vis = image.copy()
            vis[masks[best].squeeze().cpu().numpy() > 0] = [0, 255, 0]
            cv2.imwrite(os.path.join(self.output_root, f"{view}_mask.jpg"), vis)

        self._project_to_3d()
        self._export(desc)

    def _project_to_3d(self):
        full_mask = np.zeros((self.points.shape[0], 1), dtype=bool)
        for mask, view in self.masks:
            intr = np.array(self.cam_params[view]["intrinsics"])
            c2w = np.array(self.cam_params[view]["c2w"])
            uv, _, depth = project_pcd(self.points, intr, c2w)
            depth_map, _ = get_depth_map(uv, depth, *mask.shape[-2:], scale=3)
            mask3d = mask_pcd_2d(uv, mask.cpu().numpy(), 0.5, depth_map, depth, self.obj_size * 0.1)[..., None]
            full_mask = np.logical_or(full_mask, mask3d)
        self.final_mask = full_mask.squeeze()

    def _export(self, description):
        points = self.pcd.point.positions.numpy()[self.final_mask]
        seg_pcd = o3d.t.geometry.PointCloud()
        seg_pcd.point.positions = o3d.core.Tensor(points)
        if "colors" in self.pcd.point:
            colors = self.pcd.point.colors.numpy()[self.final_mask]
            seg_pcd.point.colors = o3d.core.Tensor(colors)
        o3d.t.io.write_point_cloud(os.path.join(self.output_root, "segmented.ply"), seg_pcd)
        with open(os.path.join(self.output_root, "description.txt"), "w") as f:
            f.write(description + "\n")
        print(f"[✓] Exported segmented point cloud and description.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Negative Affordance-aware Segmentation.",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息格式
    )
    parser.add_argument('-t', '--image_root', type=str, default='', required=True, description='Path to the root directory containing images.' )
    parser.add_argument('-p', '--pcd_root', type=str, default='', required=True, description='Path to the root directory containing PCD files.' )
    parser.add_argument('-o', '--output_root', type=str, default='', required=True, description='Path to the root directory for output files.' )
    parser.add_argument('-c', '--sam_ckpt', type=str, default='', required=True, description='Path to the SAM checkpoint file.' )
    args = parser.parse_args()

    IMAGE_ROOT = args.image_root
    PCD_ROOT = args.pcd_root
    OUTPUT_ROOT = args.output_root
    SAM_CKPT = args.sam_ckpt

    all_names = sorted([
        d for d in os.listdir(IMAGE_ROOT)
        if os.path.isdir(os.path.join(IMAGE_ROOT, d))
    ])

    for idx, name in enumerate(all_names):
        print(f"\n🚀 [{idx+1}/{len(all_names)}] Processing: {name}")
        try:
            task = PointCloudSegmentationTask(
                name=name,
                image_root=os.path.join(IMAGE_ROOT, name),
                pcd_root=PCD_ROOT,
                output_root=OUTPUT_ROOT,
                sam_ckpt=SAM_CKPT,
            )
            task.run()
        except Exception as e:
            print(f"[❌] Failed: {name} - {e}")
