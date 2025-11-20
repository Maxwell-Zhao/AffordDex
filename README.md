<h1 align="center"> Towards Affordance-Aware Robotic Dexterous Grasping 

with Human-like Priors </h1>

<div align="center">

This is the official repository of [**Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors**](https://afforddex.github.io/). For more information, please visit our project page.

[[Website]](https://afforddex.github.io/)
[[Arxiv]](https://arxiv.org/pdf/2508.08896)

[![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()


</div>

## Pipeline

A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordanceaware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pretrained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably humanlike in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms stateof-the-art baselines across seen objects, unseen instances, and even entirely novel categories.

<div align="center">
<img src="Figure/pipeline.png" width="520px"/>
</div>

## TODO
- [✅] Release arXiv technique report
- [✅] Release Negative Affordance-aware Segmentation pipeline
- [✅] Release Human Hand Trajectory Imitating pipeline
- [] Release Affordance-aware Residual Learning pipeline



# Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). We test with `Preview Release 3/4` and `Preview Release 4/4` version of IsaacGym and use the `Preview Release 3/4` in our paper experiment.

Please follow the steps below to perform the installation：

### 1. Create virtual environment
```python
conda create --name afforddex python=3.10 -y
conda activate afforddex
pip install -r requirements.txt
```

### 2. Install isaacgym
Once you have downloaded IsaacGym:
```bash
cd <PATH_TO_ISAACGYM_INSTALL_DIR>/python
pip install -e .
```
Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Please follow troubleshooting steps described in the Isaac Gym Preview Release 3/4
install instructions if you have any trouble running the samples.

### 3. Install dexgrasp
Once Isaac Gym is installed and samples work within your current python environment, install this repo from source code:
```bash
cd <PATH_TO_DEXGRASP_POLICY_DIR>
pip install -e .
```

### 4. Install pointnet2_ops
```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

# Dataset
We use [UniDexGrasp](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/). Addtionaly please download [datasetv4.1_posedata.npy](https://drive.google.com/file/d/1DajtOFyTPC5YhsO-Fd3Gv17x7eAysI1b/view?usp=share_link) under `assets`. Also we use [OakInk2](https://oakink.net/v2/) to train our Human Hand Trajectory Imitating.


# Training

## 1. Negative Affordance-aware Segmentation
Our NAA module provides explicit constraints on where not to touch an object. 

```python
cd NAA
python no_render_app.py --image_root xxx --pcd_root xxx --output_root xxx --sam_ckpt xxx
```

## 2. Human Hand Trajectory Imitating
We are actively working on this section and will update it soon. 🚧

## 3. Affordance-aware Residual Learning
We are actively working on this section and will update it soon. 🚧


# Citation
If you find our work useful, please consider citing us!

```bibtex
@article{zhao2025towards,
  title={Towards affordance-aware robotic dexterous grasping with human-like priors},
  author={Zhao, Haoyu and Zhuang, Linghao and Zhao, Xingyue and Zeng, Cheng and Xu, Haoran and Jiang, Yuming and Cen, Jun and Wang, Kexiang and Guo, Jiayan and Huang, Siteng and others},
  journal={arXiv preprint arXiv:2508.08896},
  year={2025}
}
```