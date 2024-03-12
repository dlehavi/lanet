import math
import random
#import itertools
import numpy as np

import torch

from model import coor_change
from model import warp_batch

def angle_axis_to_quat(angle: float, axis:list[float]):
    s = math.sin(angle / 2)
    return [math.cos(angle / 2), s * axis[0], s * axis[1], s * axis[2]]

def quat_to_rot(quat: list[float]):
    a, b, c, d = quat
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    diag = a2 - b2 - c2 - d2
    ab = 2 * a * b
    ac = 2 * a * c
    ad = 2 * a * d
    bc = 2 * b * c
    bd = 2 * b * d
    cd = 2 * c * d
    # row major
    return torch.tensor([[diag + 2 * b2, bc + ad, bd - ac],
                         [bc - ad, diag + 2 * c2, cd + ab],
                         [bd + ac, cd - ab, diag + 2 * d2]]).t()

def normalize_vec(vec:list[float]):
    v = np.array(vec)
    return list(v / np.linalg.norm(v, 2))

def one_warp_batch_test():
    # there is a default plan at z = 0
    image_H = 2 * random.randint(50, 100)
    image_W = 2 * random.randint(100, 150)
    B = random.randint(2, 3)
    source_H = random.randint(5, 7)
    source_W = random.randint(11, 13)
    # it seems we work with row major
    intrinsic = torch.tensor(
      [[image_H, 0, 0], [0, image_W, 0], [image_H / 2., image_W / 2. , 1]]).t()
    intrinsic_inv = torch.inverse(intrinsic)
    # ij is 3d, but flatenned to 2
    ij = torch.tensor([[(float(i), float(j), float(1))
                        for j in range(-image_W // 2, image_W // 2)]
                       for i in range(-image_H // 2, image_H // 2)])
    # in the line below the first two entries of the size should be image_H, image_W
    ij = ij.view(-1, 3)
    ij = torch.matmul(ij, intrinsic_inv.t())
    sources = []
    targets = []
    sources_depth = []
    targets_depth = []
    coor_change_rot = []
    coor_change_trans = []
    for _ in range(B):
        trans1 = torch.tensor(
          [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-10, -15)])
        trans2 = torch.tensor(
          [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-10, -15)])
        axis1 = normalize_vec(
          [random.uniform(-.05, .05), random.uniform(-.05, .05), 1])
        axis2 = normalize_vec(
          [random.uniform(-.05, .05), random.uniform(-.05, .05), 1])
        angle1 = random.uniform(-.2, .2)
        angle2 = random.uniform(-.2, .2)
        rot1 = quat_to_rot(angle_axis_to_quat(angle1, axis1))
        rot2 = quat_to_rot(angle_axis_to_quat(angle2, axis2))
        inv_trans1 = torch.matmul(rot1.t(), torch.mul(-1, trans1))
        inv_trans2 = torch.matmul(rot2.t(), torch.mul(-1, trans2))
        change_rot, change_trans = coor_change(rot1, trans1, rot2, trans2)
        coor_change_rot.append(change_rot)
        coor_change_trans.append(change_trans)
        for _ in range(source_H):
            for _ in range(source_W):
                pt = torch.tensor((random.uniform(-1, 1), random.uniform(-1, 1), 0))
                source_pt = torch.matmul(intrinsic, torch.addmv(inv_trans1, rot1.t(), pt))
                target_pt = torch.matmul(intrinsic, torch.addmv(inv_trans2, rot2.t(), pt))
                # TODO: should I interchange x,y here ?
                sources.append(torch.tensor(
                  [source_pt[0] / source_pt[2], source_pt[1] / source_pt[2]]))
                targets.append(torch.tensor(
                  [target_pt[0] / target_pt[2], target_pt[1] / target_pt[2]]))
        # computing the depth
        # intrinsic * pose^{-1} * pt = (ptojectively) (i : j : 1)^t
        # hence, if we set V:=intrinsic^{-1}(i, j, 1)^t = (a : b : c), and since the image is a wall at z=0
        # we have R(a', b', depth)=[?, ?, 0] - T where (a', b', depth) = depth/c (a, b, c)
        # I.e. (depth/c) RV=[?,?,0] - T or
        # (depth/c) <R_3, V> = -T_3, or depth = -cT_3 / <R_3,V>
        source_depth = -trans1[2] * torch.div(ij[:,2], torch.matmul(ij, rot1[2, :].t()))
        target_depth = -trans2[2] * torch.div(ij[:,2], torch.matmul(ij, rot2[2, :].t()))
        sources_depth.append(source_depth.view(image_H, image_W))
        targets_depth.append(target_depth.view(image_H, image_W))

    out = warp_batch(torch.stack(sources, dim=0).view(B, source_H, source_W, 2),
                   torch.stack(sources_depth, dim=0),
                   coor_change_rot, coor_change_trans, intrinsic)
    targets = torch.stack(targets, dim=0).view(B, source_H, source_W, 2)
    out = out - targets
    print('min = ', torch.min(out).item(), ', max = ', torch.max(out).item(), ', norm/size = ',
          torch.norm(out).item() / (B * source_H * source_W))
    # verify that the results are the same.

def main():
    for test in range(5):
        one_warp_batch_test()

if __name__ == '__main__':
    main()
