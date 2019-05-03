from easydict import EasyDict as edict
import dataset.zpark as zpark
import dataset.dlake as dlake
import dataset.data_iters as data_iter
import data_transform as ts
from collections import OrderedDict
import cv2
import numpy as np


config = edict()
config.path = edict()
config.path.data_root = './data/'
config.path.model_root = './models/'

config.dataset = edict()
config.dataset.zpark = zpark
config.dataset.dlake = dlake

def get_pose_cnn_setting(with_points=False,
                         K=None,
                         with_pose_in=False,
                         pre_render=True,
                         rand_num=10):
    """
    image: the image input
    label_db: rendered label map from 3D points
    pose_in: the noisy pose

    """
    data= OrderedDict([])
    data['image'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':ts.image_transform,
            'transform_params':{}}

    if pre_render:
        data['label_db'] = {'size': [512, 608],
                'reader': data_iter.trans_reader_pre_all,
                'reader_params': {'rand_num':rand_num},
                'channel': 1,
                'is_img': False,
                'resize_method': cv2.INTER_NEAREST,
                'transform': ts.label_db_transform,
                'transform_params':{'with_channel':True}}
    else:
        data['label_db'] = {'size': [512, 608],
                'reader': data_iter.trans_reader,
                'reader_params': {'multi_return': True,
                                  'proj_mat': True},
                'channel': 1,
                'is_img': False,
                'resize_method': cv2.INTER_NEAREST,
                'transform': ts.label_db_transform,
                'transform_params': {'with_channel':True}}

    if with_pose_in:
        data['pose_in'] = {'reader': data_iter.trans_reader,
                'reader_params': {'multi_return': True,
                                  'convert_to': 'mat'},
                'is_img': False,
                'transform': ts.pose_transform,
                'transform_params':{}}

    if with_points:
        data['points'] = {'size': [512/2, 608/2],
                'reader': data_iter.depth_reader,
                'is_img': False,
                'reader_params': {'K': K},
                'transform':ts.point_transform,
                'transform_params':{}}

    label = OrderedDict([])
    if pre_render:
        label['pose'] = {'reader': np.loadtxt,
                'reader_params': {},
                'transform' : ts.pose_transform,
                'transform_params': {}}

    return data, label


def get_seg_cnn_setting(with_3d=False,
                       pre_render=False,
                       method='',
                       ignore_labels=0,
                       gt_type='',
                       obj_ids=None,
                       label_mapping=None):

    """
    Inputs:
        with_3d: whether have rendered label as input for the network
        pre_render: whether directly load the pre-rendered label map
    """

    data= OrderedDict([])
    data['data'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':ts.image_transform,
            'transform_params':{}}

    if with_3d:
        if pre_render:
            data['label_db'] = {'size': [512, 608],
                    'reader': data_iter.trans_reader_pre,
                    'reader_params': {'is_gt':True} if method=='gt' \
                            else {},
                    'channel': 1,
                    'is_img': False,
                    'resize_method': cv2.INTER_NEAREST,
                    'transform':ts.label_db_transform,
                    'transform_params':{'with_channel':True}}
        else:
            data['label_db'] = {'size': [512, 608],
                    'reader': data_iter.trans_reader,
                    'reader_params': {'multi_return': False,
                                      'proj_mat': True},
                    'channel': 1,
                    'is_img': False,
                    'resize_method': cv2.INTER_NEAREST,
                    'transform':ts.label_db_transform,
                    'transform_params':{'with_channel':True,
                        'ignore_labels':ignore_labels}}

    label = OrderedDict([])
    label['softmax_label'] = {'size': [512, 608],
            'channel': 1,
            'resize_method': cv2.INTER_NEAREST,
            'transform':ts.label_transform,
            'transform_params':{
                'label_mapping':label_mapping}}

    return data, label


config.network = edict()
config.network.pose_cnn_setting = get_pose_cnn_setting
config.network.seg_cnn_setting = get_seg_cnn_setting




