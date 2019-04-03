import numpy as np
import os
import cv2
from collections import OrderedDict
from functools import wraps


def judge_exist(func):
    @wraps(func)
    def inner(*args, **kwargs):
        assert os.path.exists(args[0])
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

    params['color_num'] = len(color) + 1 # with extra background 0
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


def label_map(label, gt_type, obj_ids=None):
    """ Convert label map to different types of ground truth
    """
    if gt_type == 'full':
        return label

    elif gt_type == 'binary':
        # separate to is object or non is object, background is 1
        assert obj_ids is not None
        label_new = label.copy()
        mask = np.zeros(label.shape)
        for idx, obj_id in enumerate(obj_ids):
            label_new[label==obj_id] = 32
            mask[label==obj_id] = 1
        mask = mask == 0
        label_new[mask] = 1
        label_new[label == 0] = 0
        label = label_new

    elif gt_type == 'bkgobj':
        for idx, obj_id in enumerate(obj_ids):
            label[label==obj_id] = 32

    elif gt_type == 'bkg':
        for idx, obj_id in enumerate(obj_ids):
            label[label==obj_id] = 0
    else:
        raise ValueError('No given ground truth type')

    return label


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

    image = image.reshape((height, width, 4))
    image = image[:, :, :3]
    image = cv2.flip(image, 0)
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


    def _to_proj_mat(trans, rot, is_quater=False):

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
        label_proj.append(
                color_to_id(label, sz[0], sz[1],
                color_map,
                is_id=is_color,
                label_mapping=label_map))

        return label_proj



