from abc import ABC, abstractmethod
import numpy as np
from dataset.transform import aa_to_rotmat

from abc import ABC, abstractmethod
import os
import numpy as np

class DexHand(ABC):
    def __init__(self):
        self._urdf_path = None
        self.side = None
        self.name = None
        self.body_names = None
        self.dof_names = None
        self.hand2dex_mapping = None
        self.dex2hand_mapping = None
        self.relative_rotation = None
        self.relative_translation = np.zeros(3)
        self.contact_body_names = None
        self.bone_links = None
        self.weight_idx = None
        self.self_collision = False

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = None
        self.Ki_rot = None
        self.Kd_rot = None
        self.Kp_pos = None
        self.Ki_pos = None
        self.Kd_pos = None
        # ? <<<<<<<<<<

    @abstractmethod
    def __str__(self):
        pass

    def to_dex(self, hand_body):
        return self.hand2dex_mapping[hand_body]

    def to_hand(self, dex_body):
        return self.dex2hand_mapping[dex_body]

    @property
    def n_dofs(self):
        return len(self.dof_names)

    @property
    def n_bodies(self):
        return len(self.body_names)

    @property
    def urdf_path(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(project_root, "../../..", self._urdf_path)

    @staticmethod
    def reverse_mapping(mapping):
        reverse = {}
        for key, values in mapping.items():
            if values is None:
                continue
            for value in values:
                if value in reverse:
                    reverse[value].append(key)
                else:
                    reverse[value] = [key]

        return reverse

# <!-- 手掌 -->
# <body name="robot0:palm">

# <!-- 食指 -->
# <body name="robot0:ffknuckle">
# <body name="robot0:ffproximal">
# <body name="robot0:ffmiddle">
# <body name="robot0:ffdistal">

# <!-- 中指 -->
# <body name="robot0:mfknuckle">
# <body name="robot0:mfproximal">
# <body name="robot0:mfmiddle">
# <body name="robot0:mfdistal">

# <!-- 无名指 -->
# <body name="robot0:rfknuckle">
# <body name="robot0:rfproximal">
# <body name="robot0:rfmiddle">
# <body name="robot0:rfdistal">

# <!-- 小指 -->
# <body name="robot0:lfmetacarpal">
# <body name="robot0:lfknuckle">
# <body name="robot0:lfproximal">
# <body name="robot0:lfmiddle">
# <body name="robot0:lfdistal">

# <!-- 拇指 -->
# <body name="robot0:thbase">
# <body name="robot0:thproximal">
# <body name="robot0:thhub">
# <body name="robot0:thmiddle">
# <body name="robot0:thdistal">

class Shadow(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "shadow"
        
        self.body_names = [
            "robot0:palm",
            "robot0:ffknuckle",
            "robot0:ffproximal",
            "robot0:ffmiddle",
            "robot0:ffdistal",
            # "robot0:fftip",  # 不存在，移除或检查 URDF
            "robot0:lfmetacarpal",
            "robot0:lfknuckle",
            "robot0:lfproximal",
            "robot0:lfmiddle",
            "robot0:lfdistal",
            # "robot0:lftip",  # 不存在
            "robot0:mfknuckle",
            "robot0:mfproximal",
            "robot0:mfmiddle",
            "robot0:mfdistal",
            # "robot0:mftip",  # 不存在
            "robot0:rfknuckle",
            "robot0:rfproximal",
            "robot0:rfmiddle",
            "robot0:rfdistal",
            # "robot0:rftip",  # 不存在
            "robot0:thbase",
            "robot0:thproximal",
            "robot0:thhub",
            "robot0:thmiddle",
            "robot0:thdistal",
            # "robot0:thtip"   # 不存在
        ] 

        self.dof_names = [
            "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
            "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
            "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
            "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
            "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"
        ]
        
        self.hand2dex_mapping = {
            # 手腕部分
            "wrist": ["robot0:palm"],  # 修改为带前缀
            
            # 拇指部分
            "thumb_proximal": ["robot0:thbase", "robot0:thproximal"],
            "thumb_intermediate": ["robot0:thhub", "robot0:thmiddle"],
            "thumb_distal": ["robot0:thdistal"],
            
            # 食指部分
            "index_proximal": ["robot0:ffknuckle", "robot0:ffproximal"],
            "index_intermediate": ["robot0:ffmiddle"],
            "index_distal": ["robot0:ffdistal"],
            
            # 中指部分
            "middle_proximal": ["robot0:mfknuckle", "robot0:mfproximal"],
            "middle_intermediate": ["robot0:mfmiddle"],
            "middle_distal": ["robot0:mfdistal"],
            
            # 无名指部分
            "ring_proximal": ["robot0:rfknuckle", "robot0:rfproximal"],
            "ring_intermediate": ["robot0:rfmiddle"],
            "ring_distal": ["robot0:rfdistal"],
            
            # 小指部分
            "pinky_proximal": ["robot0:lfmetacarpal", "robot0:lfknuckle", "robot0:lfproximal"],
            "pinky_intermediate": ["robot0:lfmiddle"],
            "pinky_distal": ["robot0:lfdistal"],
        }

        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)

        self.weight_idx = {
            "thumb_tip": [],
            "index_tip": [],
            "middle_tip": [],
            "ring_tip": [],
            "pinky_tip": [],
            "level_1_joints": [1,2,6,7,10,11,14,15,18,19],  # 各手指基关节
            "level_2_joints": [3,4,5,8,9,12,13,16,17,20,21,22]  # 中间关节
        }

    def __str__(self):
        return self.name

class ShadowRH(Shadow):
    def __init__(self):
        super().__init__()
        # self._urdf_path = "assets/shadow_hand/shadow_hand_right_woarm.urdf"
        # self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0, -np.pi / 2, 0]))
        # self.relative_rotation = aa_to_rotmat(np.array([0, 0, np.pi/2]))

    def __str__(self):
        return super().__str__() + "_rh"



