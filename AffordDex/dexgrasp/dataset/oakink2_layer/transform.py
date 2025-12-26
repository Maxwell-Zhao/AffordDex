import torch


def rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        rotvec: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(rotvec, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quat = torch.cat([torch.cos(half_angles), rotvec * sin_half_angles_over_angles], dim=-1)
    return quat


def quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quat: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quat[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quat[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quat[..., 1:] / sin_half_angles_over_angles


ROT_FIELD_LIST = [
    "world_rot",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
]


def cvt_quat_to_rotvec(input_map):
    res = input_map.copy()
    for field in ROT_FIELD_LIST:
        res[field] = quat_to_rotvec(input_map[field])
    return res


def cvt_rotvec_to_quat(input_map):
    res = input_map.copy()
    for field in ROT_FIELD_LIST:
        res[field] = rotvec_to_quat(input_map[field])
    return res
