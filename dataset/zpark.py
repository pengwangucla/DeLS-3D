# preprocess the training images
import os
import cv2
import sys

python_version = sys.version_info.major
import json

import numpy as np
import utils.utils as uts
from collections import OrderedDict


def merge_pose_files(res_dir):
    """ merge all the pose file in to a single file
    """
    scenes = os.listdir(res_dir)
    print scenes
    for scene in scenes:
        path = '%s/%s/' % (res_dir, scene)
        seqs =[name for name in os.listdir(path) if os.path.isdir(path + name)]

        for seq in seqs:
            seq_file = '%s/%s/%s' % (res_dir, scene, seq)
            print seq_file
            f = open(seq_file + '.txt', 'w')
            # image_names = [name for name in os.listdir(seq_file) if '.txt' in name]
            image_names = uts.list_images(seq_file, exts=set(["txt"]))
            for image_name in image_names:
                image_name = image_name[:-4]
                pose_in_file = '%s/%s.txt' % (seq_file, image_name)
                pose_in = np.loadtxt(pose_in_file)
                pose_in = pose_in.flatten()
                res_str = image_name + '.jpg '
                idx = [3, 4, 5, 0, 1, 2]
                for i in range(6):
                    pattern = '%.4f,' if i < 5 else '%.4f\n'
                    res_str += pattern % pose_in[idx[i]]
                f.write(res_str)
            f.close()


HOME='/home/peng/Data/'

def read_depth(depth_name):
    if not os.path.exists(depth_name):
        raise ValueError("{} not exists".format(depth_name))
    if isinstance(depth_name, str):
        depth = cv2.imread(depth_name, -1)
    else:
        depth = depth_name

    depth = (1.- np.float32(depth) / 65535.0) * (300. - 0.15) + 0.15
    return depth


def read_color_label(label_name):
    if not os.path.exists(label_name):
        raise ValueError("{} not exists".format(label_name))
    label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)

    # must be color image
    if np.ndim(label) == 2:
        label = np.tile(label[:, :, None], [1, 1, 3])

    label = label[:, :, ::-1]
    return label


def hex_to_rgb(hex_str):
    return [int(hex_str[i:i+2], 16) for i in (0, 2 ,4)]


def get_config(json_file=None):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def label_map(label, gt_type, obj_ids=None):
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


def set_params(val_id=-1):
    params = {'stage': 2}
    params['data_path'] = HOME + 'zpark/'
    params['image_path'] = params['data_path'] + 'Images/'
    params['depth_path'] = params['data_path'] + 'Depth/'
    params['pose_path'] = params['data_path'] + 'Results/Loc/'
    params['label_path'] = params['data_path'] + 'LabelFull/'
    # full with manually labelled object
    params['label_color_path'] = params['data_path'] + 'LabelFullColor/'

    # path directly rendered
    params['label_bkg_path'] = params['data_path'] + 'LabelBkg/'
    # bkg with manually impainted building
    params['label_bkgfull_path'] = params['data_path'] + 'LabelBkgFull/'
    # bkg with single object foreground only
    params['label_bkgobj_path'] = params['data_path'] + 'LabelBkgObj/'

    shader_path = "/home/peng/test/baidu/personal-code/projector/src/"
    params['vertex'] = shader_path + "PointLabel.vertexshader"
    params['geometry'] = shader_path + "PointLabel.geometryshader"
    params['frag'] = shader_path + "PointLabel.fragmentshader"
    params['is_color_render'] = True

    # shader_path = "/home/peng/test/baidu/personal-code/projector/shaderQuad/"
    # params['vertex'] = shader_path + "PointLabel.vert";
    # params['geometry'] = shader_path + "PointLabel.geom";
    # params['frag'] = shader_path + "PointLabel.frag";
    params['cloud'] = "/home/peng/Data/zpark/BkgCloud.pcd";
    # params['cloud'] = "/home/peng/Data/zpark/cluster1686.pcd";

    params['label_obj_path'] = params['data_path'] + 'Label_object/0918_moving/label/'
    params['label_color_path_v1'] = params['data_path'] + 'SemanticLabel/'
    params['label_path_v1'] = params['data_path'] + 'Label/'

    params['output_path'] = params['data_path'] + 'Results/'
    params['train_set'] = params['data_path'] + 'split/train.txt'
    params['test_set'] = params['data_path'] + 'split/val.txt'

    # simulated seqences for testing
    params['sim_path'] = params['data_path'] + 'sim_test/'

    # height width
    params['size'] = [256, 304]
    params['in_size'] = [512, 608]
    params['out_size'] = [128, 152]
    params['size_stage'] = [[8, 9], [64, 76]]
    params['batch_size'] = 4
    scenes = os.listdir(params['image_path'])
    uts.rm_b_from_a(scenes, ['gt_video.avi'])
    params['scene_names'] = []
    params['camera'] = ['Camera_1', 'Camera_2']
    for scene in scenes:
        for camera in params['camera']:
            params['scene_names'].append(scene + '/' + camera)

    params['test_scene'] = []
    if val_id == -1:
        for scene in scenes[-1:]:
            for camera in params['camera']:
                params['test_scene'].append(scene + '/' + camera)

    params['train_scene'] = params['scene_names']
    uts.rm_b_from_a(params['train_scene'],
            ['Record005/Camera_1', 'Record005/Camera_2'] + params['test_scene'])
    params['intrinsic'] = {
            'Camera_1': np.array([1450.317230113, 1451.184836113, 1244.386581025, 1013.145997723]),
            'Camera_2': np.array([1450.317230113, 1451.184836113, 1244.386581025, 1013.145997723])
            }
    params['cam_names'] = params['intrinsic'].keys()

    params['raw_size'] = [2056, 2452]
    params['size'] = [x / 4 for x in params['raw_size']]

    for cam in params['intrinsic'].keys():
        params['intrinsic'][cam][[0, 2]] /= params['raw_size'][1]
        params['intrinsic'][cam][[1, 3]] /= params['raw_size'][0]

    params['read_depth'] = read_depth

    color_params = gen_color_list(params['data_path'] + 'color_v2.lst')
    params['class_num'] = color_params['color_num'] # with extra background 0
    params.update(color_params)

    params['id_2_trainid'] = np.arange(256)
    params['class_names'] = ['background',
            'sky',
            'car',
            'bike',
            'pedestrian',
            'cyclist',
            'unknown-surface',
            'car-lane',
            'pedestrian-lane',
            'bike-lane',
            'unknown',
           'unknown-road-edge',
           'curb',
           'unknown-lane-barrier',
           'traffic-cone',
           'traffic-stack',
           'fence',
           'unknown-road-side-obj',
           'light-pole',
           'traffic-light',
           'telegraph-pole',
           'traffic-sign',
           'billboard',
           'bus-stop-sign',
           'temp-building',
           'building',
           'newstand',
           'policestand',
           'unknown',
           'unknown',
           'unknown-plants',
           'plants',
           'vehicle',
           'motor-vehicle',
           'bike',
           'pedestrian',
           'cyclist']

    # these weights are use for sample points for training projective loss
    params['label_weight'] = [0, 0.1, 0, 0, 0, 0,
            1, 1, 3, 3, 0, 5, 10, 10, 10, 10, 10, 0,
            100, 100, 100, 100, 100, 10, 2, 5, 5, 5, 0, 0,
            0, 1, 0, 0, 0, 0, 0]

    params['is_rare'] = [False for i in range(params['class_num'])]
    for i in [23,26]:
        params['is_rare'][i] = True

    params['is_exist'] = [True for i in range(params['class_num'])]
    for i in [0, 2,3,4,5,10,26,28,29]:
        params['is_exist'][i] = False

    params['is_obj'] = [False for i in range(params['class_num'])]
    params['obj_ids'] = [2,3,4,5,32,33,34,35,36]
    for i in params['obj_ids']:
        params['is_obj'][i] = True

    params['is_unknown'] = [False for i in range(params['class_num'])]
    for i in range(params['class_num']):
        if 'unknown' in params['class_names'][i]:
            params['is_unknown'][i] = True

    return params


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


def eval_reader(scene_names, height, width, params):
    def get_image_list(scene_name):
        image_path = params['image_path'] + scene_name + '/'
        trans_list = uts.list_images(image_path, exts=set(['txt']))
        ext = [line for line in open(trans_list[0])]
        ext = ext[1:]
        image_name_list = []
        for line in ext:
            image_name_list.append(line.split('\t')[1][:-4])
        return image_name_list

    def reader():
        # loading path
        for scene_name in scene_names:
            res_path = params['res_path'] + scene_name + '/'
            gt_path = params['label_path'] + scene_name + '/'
            image_list = get_image_list(scene_name)
            image_num = len(image_list)
            for i, name in enumerate(image_list):
                if i % 10 == 1:
                    print "{} / {}".format(i, image_num)
                label_res = cv2.imread(res_path + name + '.png')[:, :, 0]
                label_gt = cv2.imread(gt_path + name + '.png')[:, :, 0]
                label_res = cv2.resize(label_res, (width, height),
                            interpolation=cv2.INTER_NEAREST)
                label_gt = cv2.resize(label_gt, (width, height),
                            interpolation=cv2.INTER_NEAREST)
                weight = np.ones((height,width), dtype=np.float32)
                weight = weight * np.float32(label_gt != 255)
                weight = weight * np.float32(label_gt != 0)
                weight = weight * np.float32(label_res != 0)

                # uts.plot_images({'label':label_gt, 'weight':weight})
                label_res = np.float32(label_res.flatten())
                label_gt = np.float32(label_gt.flatten())
                weight = np.float32(weight.flatten())
                yield label_res, label_gt, weight

    return reader


# This reader is for training
def reader_creator(scene_names, height, width,
                   max_num=None):
    def reader():
        pass
    return reader


def test_eval(scene_names, height, width, params):
    return eval_reader(scene_names, height, width, params)


if __name__ == "__main__":
    test_eval()
