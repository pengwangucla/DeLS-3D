# preprocess the training images
import os

import numpy as np
import utils.dels_utils as uts


def get_label_mapping(gt_type, params):
    # convert current id to training id
    label_mapping = range(256)
    if gt_type == 'full' or gt_type == 'bkgfull':
        return range(256)

    elif gt_type == 'bkg':
        for idx, obj_id in enumerate(params['obj_ids']):
            label_mapping[obj_id] = 0

    elif gt_type == 'bkgobj':
        for idx, obj_id in enumerate(params['obj_ids']):
            label_mapping[obj_id] = 32

    else:
        raise ValueError('No given ground truth type')

    # rm not evaluation classes
    for i in range(params['class_num']):
        if (not params['is_exist'][i]) \
                or params['is_rare'][i] \
                or params['is_unknown'][i]:
           label_mapping[i] = 0

    return np.array(label_mapping)


def set_params(val_id=-1):
    params = {'stage': 2}
    root_path = './'
    params['data_path'] = root_path + 'data/zpark/'
    params['image_path'] = params['data_path'] + 'images/'
    params['depth_path'] = params['data_path'] + 'projected/depth/'
    params['pose_path'] = params['data_path'] + 'camera_pose/'
    params['label_path'] = params['data_path'] + 'semantic_label/'

    # path directly rendered
    params['label_bkg_path'] = params['data_path'] + '/projected/label/'
    # bkg with inpainted background
    params['label_bkgfull_path'] = params['data_path'] + 'projected/label_inpaint/'
    params['cloud'] = params['data_path'] + "semantic_3D_point/Semantic_3D_points.pcd";

    # constructed a dummy point cloud for debugging
    # params['cloud'] = "/home/peng/Data/zpark/cluster1686.pcd";

    params['train_set'] = params['data_path'] + 'split/train.txt'
    params['test_set'] = params['data_path'] + 'split/val.txt'

    shader_path = os.environ['SHARDER']
    params['vertex'] = shader_path + "PointLabel.vertexshader"
    params['geometry'] = shader_path + "PointLabel.geometryshader"
    params['frag'] = shader_path + "PointLabel.fragmentshader"
    params['is_color_render'] = True

    # results/pose/pose_cnn or pose_rnn,  results/segments
    params['output_path'] = root_path + 'results/zpark/'

    # sample simulated seqences for testing
    params['noisy_pose_path'] = params['output_path'] + 'noisy_pose/'

    params['camera'] = ['Camera_1', 'Camera_2']
    scenes = os.listdir(params['image_path'])
    params['scene_names'] = []
    for scene in scenes:
        for camera in params['camera']:
            params['scene_names'].append(scene + '/' + camera)

    params['test_scene'] = []
    if val_id == -1:
        for scene in scenes[-1:]:
            for camera in params['camera']:
                params['test_scene'].append(scene + '/' + camera)

    # each scene forms a sequence of data points
    params['train_scene'] = uts.parse_scenes(params['train_set'])
    params['test_scene'] = uts.parse_scenes(params['test_set'])

    params['intrinsic'] = {
            'Camera_1': np.array([1450.317230113, 1451.184836113,
                                  1244.386581025, 1013.145997723]),
            'Camera_2': np.array([1450.317230113, 1451.184836113,
                                  1244.386581025, 1013.145997723])
            }
    params['cam_names'] = params['intrinsic'].keys()
    params['raw_size'] = [2056, 2452]
    for cam in params['intrinsic'].keys():
        params['intrinsic'][cam][[0, 2]] /= params['raw_size'][1]
        params['intrinsic'][cam][[1, 3]] /= params['raw_size'][0]

    # for network configuration height width
    params['out_size'] = [128, 152]
    params['size_stage'] = [[8, 9], [64, 76]] # for training
    params['size'] = [512, 608]
    params['batch_size'] = 4

    params['read_depth'] = uts.read_depth

    color_params = uts.gen_color_list(params['data_path'] + 'color_v2.lst')
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

    # whether the semantic label appears in test set
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



if __name__ == "__main__":
    set_params()
