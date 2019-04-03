# preprocess the training images
import os
import sys
import zpark
import labels

python_version = sys.version_info.major
import numpy as np
from collections import OrderedDict


def set_params(val_id=-1):
    params = OrderedDict([])
    params['data_path'] = './data/dlake/'
    params['image_path'] = params['data_path'] + 'image/'
    params['depth_path'] = params['data_path'] + 'depth/'
    params['pose_path'] = params['data_path'] + 'camera_pose/'
    params['pose_mat_path'] = params['data_path'] + 'pose/'

    params['label_path'] = params['data_path'] + 'label/' # has not mapped
    params['label_bkg_path'] = params['data_path'] + 'label_render/' # mapped
    params['label_bkgfull_path'] = params['data_path'] + 'label_render/' # mapped inpainted
    params['label_color_path'] = params['data_path'] + 'label_color/'# mapped

    shader_path = "/home/wangp/baidu/personal-code/projector/"
    params['vertex'] = shader_path + "shaderQuadV2/PointLabel_quad_gray.vert"
    params['geometry'] = shader_path + "shaderQuadV2/PointLabel_quad.geom"
    params['frag'] = shader_path + "shaderQuadV2/PointLabel_quad.frag"
    params['is_color_render'] = False

    params['cloud'] = params['data_path'] + "mergedBkg.pcd";
    # params['cloud'] = params['data_path'] + "cluster1686.pcd";

    # simulated seqences for testing
    params['output_path'] = params['data_path'] + 'Results/'
    params['pose_permute'] = params['output_path'] + 'permute/'
    params['pose_permute_num'] = 10
    params['sim_path'] = params['output_path'] + 'sim_test/'
    params['pose_rect'] = params['output_path'] + 'pose_rect/'

    params['train_set'] = params['data_path'] + 'split/train.txt'
    params['test_set'] = params['data_path'] + 'split/val.txt'

    # height width
    params['in_size'] = [512, 608]
    params['out_size'] = [128, 152]
    params['batch_size'] = 1

    scenes = os.listdir(params['image_path'])
    params['intrinsic'] = {
            'Camera 5' : np.array([2304.54786556982, 2305.875668062, 1686.23787612802, 1354.98486439791]),
            'Camera 6': np.array([2300.39065314361, 2301.31478860597, 1713.21615190657, 1342.91100799715])
            }
    params['cam_names'] = params['intrinsic'].keys()

    params['raw_size'] = [2710, 3384]
    for c_name in params['intrinsic'].keys():
        params['intrinsic'][c_name][[0, 2]] /= params['raw_size'][1]
        params['intrinsic'][c_name][[1, 3]] /= params['raw_size'][0]

    params['read_depth'] = zpark.read_depth

    color_params = zpark.gen_color_list(params['data_path'] + 'color.lst')
    params['class_num'] = color_params['color_num'] # with extra background 0
    params.update(color_params)

    # mapping between saved id and trainig id
    params['id_2_trainid'] = np.full((256,), 0)
    for label in labels.labels:
        if label.id in (-1,):
            continue
        params['id_2_trainid'][label.id] = label.trainId

    params['class_names'] = [
           'background',
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
           'wall',
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
           'motor-vehicle',
           'truck',
           'bus',
           'triclist',
           'none']

    # these weights are use for sample points for training projective loss
    params['label_weight'] = [0, 0.1, 0, 0, 0, 0, 1, 1, 3, 3, 0, 5, 10, 10, 10, 10, 10, 0,
                              100, 100, 100, 100, 100, 10, 2, 5, 5, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    params['is_rare'] = [False for i in range(params['class_num'])]
    for i in [23,26]:
        params['is_rare'][i] = True

    params['is_exist'] = [True for i in range(params['class_num'])]
    for i in [0,2,3,4,5,10,26,28,29]:
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


def map_class_id():
    map_id = {}
    return map_id


if __name__ == "__main__":
    set_params()
