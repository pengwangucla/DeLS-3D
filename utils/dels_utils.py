"""
Utils for dels 3D with apolloscape dataset
Author: 'peng wang'

"""

import numpy as np
import os
import cv2
from collections import OrderedDict
from functools import wraps

def judge_exist(func):
    @wraps(func)
    def inner(*args, **kwargs):
        assert os.path.exists(args[0]), args[0]
        return func(*args, **kwargs)
    return inner

@judge_exist
def imread(image_file):
    return cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

@judge_exist
def loadtxt(pose_file):
    return np.loadtxt(pose_file)


@judge_exist
def read_depth(depth_name):
    """
    Beta version depth saving format for Zpark
    """
    if isinstance(depth_name, str):
        depth = cv2.imread(depth_name, -1)
    else:
        depth = depth_name

    depth = (1.- np.float32(depth) / 65535.0) * (300. - 0.15) + 0.15
    return depth


# reweighting different labels
def label2weight(label, weights):
    ids = np.unique(label)
    weight = np.zeros_like(label)
    for idx in ids:
        weight[label == idx] = weights[idx]

    return weight


def point_reader(depth_file,
                 K,
                 sz=None,
                 label_file=None,
                 label_weight=False):
    """
       Reading 3D points from a depth map
    Inputs:
        depth_file: depth image
        K: intrinsic
    """

    import utils_3d as uts_3d
    depth = read_depth(depth_file)
    if sz is not None:
        depth = cv2.resize(depth, (sz[1], sz[0]),
                interpolation=cv2.INTER_NEAREST)

    height, width = depth.shape
    points = uts_3d.depth2xyz(depth, K)

    if label_file:
        label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        weight_map = label2weight(label, label_weight)
        weight_mask = cv2.resize(weight_map,
                (width, height), interpolation=cv2.INTER_NEAREST)
        index = weight_mask > 0
        index = index.flatten()
        points = points[index, :]
        weight_mask = weight_mask.flatten()[index]

        return points, weight_mask

    weight_mask = np.ones(sz[1] * sz[0], dtype=np.float32)
    return points, weight_mask


def hex_to_rgb(hex_str):
    return [int(hex_str[i:i+2], 16) for i in (0, 2 ,4)]


def read_color_label(label_name):
    if not os.path.exists(label_name):
        raise ValueError("{} not exists".format(label_name))
    label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)

    # must be color image
    if np.ndim(label) == 2:
        label = np.tile(label[:, :, None], [1, 1, 3])

    label = label[:, :, ::-1]
    return label


def gen_color_list(color_file):
    params = {}
    color = [line for line in open(color_file)]
    params['color_map'] = OrderedDict([((255, 255, 255), 0)])

    for i, line in enumerate(color):
        hex_color = line.split('\t')[0]
        rgb = tuple(hex_to_rgb(hex_color[2:]))
        params['color_map'][rgb] = i + 1

    params['color_map_list'] = []
    for line in color:
        hex_color = line.split('\t')[0]
        params['color_map_list'].append(hex_to_rgb(hex_color[2:]))

    # with extra background 0
    params['color_num'] = len(color) + 1

    return params


def parse_scenes(apollo_file):
    lines = [x.split('/')[0] for x in open(apollo_file, 'r')]
    unique_lines = [lines[0]]
    for i in range(1, len(lines)):
        if lines[i] == lines[i-1]:
            continue
        else:
            unique_lines.append(lines[i])
    return unique_lines


def get_config(json_file=None):
    import json
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def image_set_2_seqs(image_set,
                     cameras,
                     max_len=None):
    """
        Convert the set of images from apolloscape records
        to video sequences based on image name
    """

    def is_conseq_frame(name, last_name):
        record = name.split('/')[0]
        record_id = int(record[-3:])
        record_last = last_name.split('/')[0]
        record_last_id = int(record_last[-3:])
        if record_id in [record_last_id, record_last_id + 1]:
            return True
        return False

    cam_seqs = []
    for Camera in cameras:
        cam_seqs.append([line for line in image_set \
                              if Camera in line])
    res_seqs = []
    for seq in cam_seqs:
        i_last = 0
        for i in range(len(seq)):
            if not is_conseq_frame(seq[i], seq[max(i-1, 0)]):
                res_seqs.append(seq[i_last:i])
                i_last = i
        if len(seq) > i_last:
            res_seqs.append(seq[i_last:len(seq)])


    # split the array with maximum len
    if max_len:
        pass

    return res_seqs


def get_intr_key(keys, filename):
    """
       get the correct intrinsic for each image.
    """
    intr_key = keys[0]
    found = False
    for key in keys:
        if key in filename:
            intr_key = key
            found = True
            break

    if not found:
        raise ValueError('Image not assigned to a intrinsic')

    return intr_key


# renderer
import renderer.projector as pj
import utils_3d as uts_3d
import vis_utils as v_uts

def color_to_id(image, height, width,
                color_map=None,
                is_id=True, # means is_color
                label_mapping=None):
    """Convert a rendered color image to id
    Current it is very confusion, i have two version render
    1: render color  2: direct render label id
    is_id: using color else use label id

    Input:
       color_map: the color map
       is_id: whether to return semantic id map rather than color
              map
       label_mapping: mapping label to used label id
    """

    label = image[:, :, ::-1]
    if is_id:
        label = v_uts.color2label(label, color_map)
    else:
        label = label[:, :, 0]
        if label_mapping is not None:
            label = label_mapping[label]

    return label



class Renderer(object):
    def __init__(self, params, cloud_name, image_size):

        self.image_size = image_size
        assert image_size[0] % 4 == 0
        assert image_size[1] % 4 == 0

        self.proj = pj.pyRenderPCD(
                        cloud_name,
                        params['vertex'],
                        params['geometry'],
                        params['frag'],
                        self.image_size[0],
                        self.image_size[1])


    def _to_proj_mat(self, trans, rot, is_quater=False):

        ext = np.zeros((4, 4), dtype=np.float32)
        if not is_quater:
            ext[:3, :3] = uts_3d.euler_angles_to_rotation_matrix(rot)
        else:
            ext[:3, :3] = uts_3d.quater_to_rot_mat(rot)

        ext[:3, 3] = trans
        ext[3, 3] = 1.0
        ext = np.linalg.inv(ext)
        ext = np.transpose(ext)
        ext = np.float32(ext.flatten())

        return ext


    def _to_proj_intr(self, intr):
        """convert 4 dim intrinsic to projection matrix
        """
        intrinsic = uts_3d.intrinsic_vec_to_mat(intr, self.image_size)
        intr_for_render = np.transpose(intrinsic)
        intr_for_render = intr_for_render.flatten()
        return intr_for_render


    def render_from_3d(self,
                       pose_in,
                       intr,
                       color_map,
                       is_color,
                       label_map):
        """
        Inputs:
            pose_in: a input pose with [n x 6] 6 dim vector,
                first 3 is trans and last 3 is rot
        return:
            a list of rendered resutls
        """

        label_proj = []
        sz = self.image_size
        ext = self._to_proj_mat(
                pose_in[:3], pose_in[3:], is_quater=False)
        intr = self._to_proj_intr(intr)
        label, depth = self.proj.pyRenderToRGBDepth(intr, ext)
        label_proj = color_to_id(label, sz[0], sz[1],
                               color_map,
                               is_id=is_color,
                               label_mapping=label_map)

        return label_proj



