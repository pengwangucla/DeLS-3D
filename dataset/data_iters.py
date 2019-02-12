"""Batch size 1 reader
   Update data iter with 3D projection
"""
# pylint: skip-file
import pdb
import mxnet as mx
import numpy as np
import os
import cv2

import data.zpark as zpark
import transform as ts
from imgaug import augmenters as iaa
from mxnet.io import DataIter, DataBatch
from collections import OrderedDict, namedtuple Batch = namedtuple('Batch', ['data']) import Networks.net_util as nuts
import Networks.pose_net as posenet
import utils.utils_3d as uts_3d
import utils.utils as uts
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

NEW_X = np.newaxis


def get_video_sequence(image_path):
    trans_list = uts.list_images(image_path, exts=set(['txt']))
    ext = [line for line in open(trans_list[0])]
    return ext[1:]


def gen_file_list(params,
                  save_path,
                  split,
                  ext='',
                  with_3d=False,
                  by_scene=False,
                  gt_type='bkg'):

    save_file = save_path + split + ext + '.ls'
    if os.path.exists(save_file):
        os.remove(save_file)

    f = open(save_file, 'a')
    prefix_image = params['image_path'].split('/')[-2] + '/'
    if with_3d:
        prefix_label_db = params['motion_path'].split('/')[-2] + '/'
    if gt_type == 'bkg':
        prefix_label = params['label_bkgfull_path'].split('/')[-2] + '/'
    elif gt_type == 'full':
        prefix_label = params['label_path'].split('/')[-2] + '/'
    else:
        prefix_label = params['label_bkgobj_path'].split('/')[-2] + '/'

    set_name = split + '_set'
    image_list = [line.strip() for line in open(params[set_name])]
    # shorten the test
    if split == 'test':
        image_list = image_list[:387]

    for filename in image_list:
        str_img = prefix_image + filename + '.jpg'
        str_lb = prefix_label + filename + '.png'
        if with_3d:
            str_loc = prefix_label_db + filename + '.txt'
            save_str = '{}\t{}\t{}\n'.format(str_img, str_loc, str_lb)
        else:
            save_str = '{}\t{}\n'.format(str_img, str_lb)
        f.write(save_str)
    f.close()


def gen_file_list_video(params, recur_steps, save_path, split):
    save_file = save_path + split + '_video_ndb' + str(recur_steps) +'.ls'
    if os.path.exists(save_file):
        os.remove(save_file)

    f = open(save_file, 'a')
    prefix_image = params['image_path'].split('/')[-2] + '/'
    prefix_label = params['label_path'].split('/')[-2] + '/'
    prefix_flow = params['output_path'].split('/')[-2] + '/SeqFlow' + '/'

    key = split.split('/')[-1] + '_scene'
    for scene_name in params[key]:
        image_path = params['image_path'] + scene_name + '/'
        image_list = get_video_sequence(image_path)
        image_num = len(image_list)

        for i in range(image_num - recur_steps + 1):
            # write a line of sequence for input
            for j in range(recur_steps):
                indice = i + j
                ext_i = image_list[indice].split('\t')
                image_name = ext_i[1]
                str_img = prefix_image + scene_name + '/' + image_name
                save_str = '{}\t'.format(str_img)
                f.write(save_str)

            # label_gt
            for j in range(recur_steps):
                indice = i + j
                ext_i = image_list[indice].split('\t')
                image_name = ext_i[1]

                if j > 0:
                    # flow
                    str_lb = prefix_flow + scene_name + '/' + image_name[:-4] + '.npz'
                    append = '\t' if j == recur_steps-1 else '\t'
                    save_str = ('{}' + append).format(str_lb)
                    f.write(save_str)

                # semantic label
                str_lb = prefix_label + scene_name + '/' + image_name[:-4] + '.png'
                append = '\n' if j == recur_steps-1 else '\t'
                save_str = ('{}' + append).format(str_lb)
                f.write(save_str)

    f.close()


def gen_flow_file_list(params, save_path, split):
    save_file = save_path + split + '_flow.ls'
    if os.path.exists(save_file):
        os.remove(save_file)

    f = open(save_file, 'a')
    prefix_image = params['image_path'].split('/')[-2] + '/'
    key = split + '_scene'
    for scene_name in params[key]:
        output_path = params['output_path'] + 'SeqFlow' + '/' + scene_name + '/'
        flow_list = uts.list_images(output_path, exts=set(['npz']))

        for flow_name in flow_list:
            image_name, image_name2 = flow_name.split('^')
            image_name = image_name.split('/')[-1]
            str_img = prefix_image + scene_name + '/' + image_name + '.jpg'
            image_name2 = image_name2.replace('-', '/')
            str_img2 = prefix_image + image_name2[:-4] + '.jpg'
            flow_name = flow_name.split('/')[-5:]
            flow_name = '/'.join(flow_name)
            save_str = '{}\t{}\t{}\n'.format(str_img, str_img2, flow_name)
            f.write(save_str)
    f.close()


def gen_pose_file_list(params, save_path, split, list_type='pose'):
    save_file = save_path + split + '_' + list_type + '.ls'
    if os.path.exists(save_file):
        os.remove(save_file)

    f = open(save_file, 'a')
    prefix_image = params['image_path'].split('/')[-2] + '/'
    prefix_pose = params['pose_path'].split('/')[-2] + '/'

    prefix_pre_render = params['pose_permute'][len(params['data_path']):]
    prefix_depth = params['depth_path'].split('/')[-2] + '/'
    prefix_label = params['label_path'].split('/')[-2] + '/'
    set_name = split + '_set'
    image_list = [line.strip() for line in open(params[set_name])]

    for filename in image_list:
        str_img = prefix_image + filename + '.jpg'
        str_pose = prefix_pose + filename + '.txt'

        if 'pose_proj' == list_type:
            depth_str = prefix_depth + filename + '.png'
            save_str = '{}\t{}\t{}\n'.format(str_img, str_pose, depth_str)

        elif 'pose_sem_proj' == list_type:
            depth_str = prefix_depth + filename + '.png'
            label_str = prefix_label + filename + '.png'
            save_str = '{}\t{}\t{}\t{}\n'.format(str_img, str_pose, depth_str, label_str)

        elif 'pose' == list_type:
            save_str = '{}\t{}\n'.format(str_img, str_pose)

        elif 'pose_seg' == list_type:
            label_str = prefix_label + filename + '.png'
            save_str = '{}\t{}\t{}\n'.format(str_img, str_pose, label_str)

        elif 'pose_pre_render_proj' == list_type:
            permute_pose = prefix_pre_render + filename + '.png'
            depth_str = prefix_depth + filename + '.png'
            save_str = '{}\t{}\t{}\t{}\n'.format(str_img, permute_pose, depth_str, str_pose)

        elif 'pose_pre_render' == list_type:
            trans_perturb_name = prefix_pre_render + filename + '.png'
            label_str = prefix_label_db + filename + '.txt'
            save_str = '{}\t{}\t{}\n'.format(str_img, trans_perturb_name, label_str)

        f.write(save_str)

    f.close()


def image_set_2_seqs(image_set, cameras, max_len=None, rand_idx=None):

    def is_conseq_frame(name, last_name):
        record = name.split('/')[0]
        record_id = int(record[-3:])
        record_last = last_name.split('/')[0]
        record_last_id = int(record_last[-3:])
        if record_id in [record_last_id, record_last_id + 1]:
            return True
        return False

    if rand_idx is None:
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

    else:
        assert(len(image_set) == len(rand_idx))
        cam_seqs = []
        for Camera in cameras:
            cam_seqs.append([(line, idx) for line, idx in  \
                             zip(image_set, rand_idx) \
                             if Camera in line])
        pdb.set_trace()
        res_seqs = []
        for seq in cam_seqs:
            i_last = 0
            for i in range(len(seq)):
                if not is_conseq_frame(seq[i][0], seq[max(i-1, 0)][0]):
                    res_seqs.append(seq[i_last:i])
                    i_last = i
            # if len(seq) > i_last:
            #     res_seqs.append(seq[i_last:len(seq)])

    # split the array with maximum len
    if max_len:
        pass

    return res_seqs


def gen_pose_rnn_file_list(params, rnn_num, save_path, split, ext,
                           pre_gen=False,
                           rect_net=None,
                           list_type='pose'):
    """
    Inputs:
        rnn_num: how long is each sequence is
        sim_num: how many simulations for each rnn sequence
    """

    save_file = save_path + split + '_' + ext + '.ls'
    if os.path.exists(save_file):
        os.remove(save_file)
    f = open(save_file, 'a')

    if pre_gen:
        key, sub = ['pose_rect', rect_net + '/'] if rect_net else \
                ['pose_permute', '']
        prefix_pose_in = params[key][len(params['data_path']):] + sub
        prefix_pose = params['pose_path']

    else:
        # we need to random perturb ground truth pose
        prefix_pose_in = params['pose_path']
        prefix_pose = params['pose_path']

    prefix_depth = params['depth_path'].split('/')[-2] + '/'
    set_name = split + '_set'

    # seqone needs to be in the same sequence
    # split the seq in each camera
    image_list = [line.strip() for line in open(params[set_name])]
    cameras = params['cam_names'] if split == 'train' else params['cam_names'][0]
    image_seq = image_set_2_seqs(image_list, params['cam_names'])
    logging.info('%s seqs len %d' % (set_name, len(image_seq)))

    for image_list in image_seq:
        seq_len = len(image_list)
        for i in range(0, seq_len - rnn_num / 2, 2):
            save_str = []
            for j in range(rnn_num):
                filename = image_list[min(i + j, seq_len-1)]
                save_str.append(prefix_pose_in + filename + '.txt')

            if 'pose_proj' == list_type:
                for j in range(rnn_num):
                    filename = image_list[min(i + j, seq_len-1)]
                    save_str.append(prefix_depth + filename + '.png')

            for j in range(rnn_num):
                filename = image_list[min(i + j, seq_len-1)]
                save_str.append(prefix_pose + filename + '.txt')

            save_str = '\t'.join(save_str) + '\n'
            f.write(save_str)
    f.close()


def gen_file_pair_list(params, save_path, split, start_id=1):
    save_file = save_path + split + '.ls'
    if os.path.exists(save_file):
        os.remove(save_file)

    f = open(save_file, 'a')
    prefix_image = params['image_path'].split('/')[-2] + '/'
    prefix_label = params['label_path'].split('/')[-2] + '/'
    i = start_id
    key = split + '_scene'
    for scene_name in params[key]:
        image_path = params['image_path'] + scene_name + '/'
        image_list = uts.list_images(image_path)

        for image_name in image_list:
            image_name = image_name.split('/')[-1]
            str_1 = prefix_image + scene_name + '/' + image_name
            str_2 = prefix_label + scene_name + '/' + image_name[:-4] + '.png'
            save_str = '{}\t{}\t{}\n'.format(i, str_1, str_2)
            i += 1
            f.write(save_str)
    f.close()
    return i


def exp_to_4d(data, shape=None, interp=None):
    if shape is not None:
        data = cv2.resize(data, (shape[1], shape[0]),
                          interpolation=interp)
    if data.ndim == 3:
        data = data[NEW_X, :, :, :]
    elif data.ndim == 2:
        data = data[NEW_X, :, :, NEW_X]

    return data


def image_transform(img, method='norm', center_crop=None):
    if 'norm' == method:
        img = np.float32(img) / 255.0
        img -= 0.5

    if center_crop:
        height, width = img.shape[:2]
        up = height / 2 - center_crop / 2
        left = width / 2 - center_crop / 2
        img = img[up:up+center_crop, left:left+center_crop]

    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)  # (1, c, h, w)
    return img


def score_transform(score):
    score = np.transpose(score, [2, 0, 1])
    score = np.expand_dims(score, axis=0)  # (1, c, h, w)
    return score


def mask_label(label, ignore_labels):
    if isinstance(ignore_labels, int):
        ignore_labels = [ignore_labels]
    mask = label >= 0
    for ignore_label in ignore_labels:
        mask = np.logical_and(mask, label != ignore_label)
    label = label * np.int32(mask)
    return label


def label_db_transform(label, with_channel=False,
        ignore_labels=None):
    #label = np.float32(label)/class_num
    if ignore_labels is not None:
        label=mask_label(label, ignore_labels)


    label = label[NEW_X, NEW_X, :, :] if with_channel \
            else label[NEW_X, :, :]

    return label # (1, h, w)


def label_transform(label,
                    ignore_labels=None,
                    gt_type=None,
                    obj_ids=None,
                    label_mapping=None):

    if label_mapping is not None:
        label = np.uint8(label)
        label = np.float32(label_mapping[label])

    if ignore_labels is not None:
        label = mask_label(label, ignore_labels)

    if gt_type is not None:
        label = zpark.label_map(label, gt_type, obj_ids)

    return  label[NEW_X, :, :]


def pose_transform(pose, mean_pose=None, scale=None,
        to_quater=False):

    if mean_pose is None:
        mean_pose = np.zeros(3, dtype=np.float32)
    if scale is None:
        scale = 1.0

    pose[:3] = (pose[:3] - mean_pose)/scale
    if to_quater:
        pose = np.concatenate([pose[:3],
            uts_3d.euler_angles_to_quaternions(pose[3:])])
    return pose[NEW_X, :]


def point_transform(points):
    points = np.transpose(points, [1, 0])
    return np.expand_dims(points, axis=0)


def flow_transform(flow):
    flow = np.transpose(flow, [2, 0, 1])
    return np.expand_dims(flow, axis=0) #(1, 2, h, w)


def identity(data):
    return data


def flow_reader(flow_file):
    res = np.load(flow_file)['flow']
    return res


def angle_to_proj_mat(trans, rot, is_quater=False):

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
        label = uts.color2label(label, color_map)
    else:
        label = label[:, :, 0]
        if label_mapping is not None:
            label = label_mapping[label]

    return label


def get_projector(params, height, width):
    import projector as pj
    proj = pj.pyRenderPCD(
            params['cloud'],
            params['vertex'],
            params['geometry'],
            params['frag'],
            height, width)
    return proj


def get_proj_intr(intr, height, width):
    intrinsic = uts_3d.intrinsic_vec_to_mat(
            intr, [height, width])
    intr_for_render = np.transpose(intrinsic)
    intr_for_render = intr_for_render.flatten()

    return intr_for_render


def get_intr_key(keys, filename):
    """get the correct intrinsic for each image.
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


def trans_reader_pre(trans_file, ext='png', is_gt=False,
        get_rand_id=False, max_pre_render_num=50):
    """read from pre rendered images
    """

    image_name = trans_file[:-4]
    if is_gt:
        label_name = image_name + '.' + ext
    else:
        rand_int = np.random.randint(max_pre_render_num)
        label_name = (image_name + "_%04d." + ext) % rand_int

    if not os.path.exists(label_name):
        raise ValueError('{} not exist'.format(label_name))

    if 'png' == ext:
        label_db = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)
    elif 'txt' == ext:
        label_db = np.loadtxt(label_name)
    else:
        raise ValueError('no such label file\n')

    if get_rand_id:
        return label_db, rand_int
    else:
        return label_db


def trans_reader_pre_all(trans_file, rand_num):

    # motion
    image_name = trans_file[:-4]
    rand_int = np.random.randint(rand_num)

    motion_noisy = np.loadtxt((image_name + "_%04d.txt") % rand_int)
    label_db = cv2.imread((image_name + "_%04d.png") % rand_int,
            cv2.IMREAD_UNCHANGED)
    label_db_file = (image_name + "_%04d.png") % rand_int
    if not os.path.exists(label_db_file):
        raise ValueError('%s not exists' % label_db_file)

    return motion_noisy, label_db


def trans_reader(trans_file,
                 convert_to='mat',
                 multi_return=False,
                 proj_mat=False):
    """ read extrinsic parameter of each image
    """

    motion = np.loadtxt(trans_file)

    # parts = trans_file.split('/')
    # image_name = '/home/wangp/data/dlake/Results/permute/' + '/'.join(parts[-3:])[:-4]
    # rand_int = 0
    # motion_noisy = np.loadtxt((image_name + "_%04d.txt") % rand_int)
    # trans_i = motion_noisy[:3]
    # rot_i = motion_noisy[3:]

    trans_i, rot_i = uts_3d.random_perturb(
            motion[:3], motion[3:], 5.0)
    motion_noisy = np.concatenate([trans_i, rot_i])

    ext = None
    ext_gt = None
    if proj_mat:
        ext = angle_to_proj_mat(trans_i, rot_i)
        ext_gt = angle_to_proj_mat(motion[:3], motion[3:])

    if 'qu' == convert_to:
        motion_noisy = np.concatenate([trans_i,
            uts_3d.euler_angles_to_quaternions(rot_i)])
        motion = np.concatenate([motion[:3],
            uts_3d.euler_angles_to_quaternions(motion[3:])])

    if multi_return:
        return motion, motion_noisy, ext, ext_gt
    else:
        return ext


def pose_reader_4rnn(pose_file,
                     proj,
                     params,
                     posecnn):
    """This is the pose reader for rnn input generation
       It needs to refine from noisy pose with posecnn
    Inputs:
        noise_pose_file:
        proj:
        params:
        posecnn:
    """

    height, width = proj.get_image_size()
    filename = pose_file[len(params['pose_path']):-4]
    image_file = params['image_path'] + filename + '.jpg'
    pose = np.loadtxt(pose_file)
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (width, height))

    trans_i, rot_i = uts_3d.random_perturb(
                       pose[:3], pose[3:], 5.0)
    pose_in = np.concatenate([trans_i, rot_i])

    ext = angle_to_proj_mat(trans_i, rot_i)
    intr_key = get_intr_key(params['intrinsic'].keys(), filename)
    intr = get_proj_intr(params['intrinsic'][intr_key], height, width)
    label_db = proj.pyRenderToMat(intr, ext)
    label_db = color_to_id(label_db,
                           height, width,
                           color_map=None,
                           is_id=False,
                           label_mapping=params['id_2_trainid'])

    pose_np = posecnn.infer(image, label_db, pose_in)
    return pose_np


def label2weight(label, weights):
    ids = np.unique(label)
    weight = np.zeros_like(label)
    for idx in ids:
        weight[label == idx] = weights[idx]

    return weight


def depth_reader(depth_file,
                 K,
                 sz=None,
                 label_file=None,
                 label_weight=False):

    # pdb.set_trace()
    depth = zpark.read_depth(depth_file)
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


def get_setting():
    data= OrderedDict([])
    data['data'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}

    data['label_db'] = {'size': [512, 608],
            'channel': 1,
            'is_img': False,
            'resize_method': cv2.INTER_NEAREST,
            'transform':label_db_transform,
            'transform_params':{'with_channel': False}}

    label = OrderedDict([])
    label['softmax_label'] = {
            'size': [128, 152],
            'channel': 1,
            'resize_method': cv2.INTER_NEAREST,
            'transform': label_transform,
            'transform_params': {}}
    return data, label


def get_pose_setting(with_points=False, K=None,
        with_pose_in=False, pre_render=True,
        rand_num=10):
    data= OrderedDict([])
    data['image'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}

    if pre_render:
        data['label_db'] = {'size': [512, 608],
                'reader': trans_reader_pre_all,
                'reader_params': {'rand_num':rand_num},
                'channel': 1,
                'is_img': False,
                'resize_method': cv2.INTER_NEAREST,
                'transform':label_db_transform,
                'transform_params':{'with_channel':True}}
    else:
        data['label_db'] = {'size': [512, 608],
                'reader': trans_reader,
                'reader_params': {'multi_return': True,
                                  'proj_mat': True},
                'channel': 1,
                'is_img': False,
                'resize_method': cv2.INTER_NEAREST,
                'transform': label_db_transform,
                'transform_params': {'with_channel':True}}

    if with_pose_in:
        data['pose_in'] = {'reader': trans_reader,
                'reader_params': {'multi_return': True,
                                  'convert_to': 'mat'},
                'is_img': False,
                'transform': pose_transform,
                'transform_params':{}}

    if with_points:
        data['points'] = {'size': [512/2, 608/2],
                'reader': depth_reader,
                'is_img': False,
                'reader_params': {'K': K},
                'transform':point_transform,
                'transform_params':{}}

    label = OrderedDict([])
    if pre_render:
        label['pose'] = {'reader': np.loadtxt,
                'reader_params': {},
                'transform' : pose_transform,
                'transform_params': {}}

    return data, label


def get_pose_rnn_setting(rnn_num,
                         with_points=False,
                         params=None,
                         with_pose_in=False,
                         is_quater=True,
                         pre_gen=False,
                         pose_cnn_model=None,
                         ctx=None):

    data= OrderedDict([])
    if pre_gen:
        for i in range(rnn_num):
            data['pose_in_%02d'%i] = {'reader': trans_reader_pre,
                        'reader_params': {'ext': 'txt',
                            'max_pre_render_num': params['pose_permute_num']},
                        'is_img': False,
                        'transform': pose_transform,
                        'transform_params' : {'to_quater':is_quater}}
    else:
        height, width = 512, 608
        # define projector
        proj = get_projector(params, height, width)

        # define posecnn
        import Networks.infer_networks as network
        posecnn = network.PoseCNN(pose_cnn_model, ctx, [height, width])

        for i in range(rnn_num):
            data['pose_in_%02d'%i] = {'reader': pose_reader_4rnn,
                        'reader_params': {'proj': proj,
                                          'params': params,
                                          'posecnn': posecnn},
                        'is_img': False,
                        'transform': pose_transform,
                        'transform_params': {'to_quater':is_quater}}


    if with_points:
        for i in range(rnn_num):
            data['points_%02d'%i] = {'size': [512/4, 608/4],
                    'reader': depth_reader,
                    'is_img': False,
                    'reader_params': {},
                    'resize_method': cv2.INTER_NEAREST,
                    'transform':image_transform,
                    'transform_params': {'method': None}}

    label = OrderedDict([])
    for i in range(rnn_num):
        label['pose_%02d'%i] = {'reader': trans_reader_pre,
                'reader_params': {'ext': 'txt',
                                  'is_gt': True},
                    'is_img': False,
                    'transform': pose_transform,
                    'transform_params':{'to_quater':is_quater}}

    return data, label


def get_pose_seg_setting(params=None):
    data= OrderedDict([])
    data['image'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}

    data['pose_in'] = {'reader': trans_reader,
            'reader_params': {'multi_return': True,
                              'convert_to': 'mat'},
            'is_img': False,
            'transform': pose_transform,
            'transform_params':{}}

    label = OrderedDict([])
    label['softmax_label'] = {
            'size': [512, 608],
            'channel': 1,
            'resize_method': cv2.INTER_NEAREST,
            'transform': label_transform,
            'transform_params': {}}

    return data, label


def get_posenet_setting(with_gps=False):
    data= OrderedDict([])
    label = OrderedDict([])

    data['image'] = {'size': [256, 304],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{'center_crop': 224}}

    if with_gps:
        data['label_db'] = {'reader': trans_reader,
                'reader_params': {'multi_return': True,
                                  'convert_to': 'qu'},
                'is_img': False,
                'transform': pose_transform,
                'transform_params':{}}
    else:
        mean_pose = np.array([
            4.4996e+02, -2.2512e+03, 4.0171e+01,
            -1.7951e+00, -2.6040e-02, -1.1490e-01])

        label['pose'] = {'reader': trans_reader,
                'reader_params': {'multi_return': True,
                                  'convert_to': 'qu'},
                'channel': 1,
                'transform': pose_transform,
                'transform_params':{'mean_pose':mean_pose[:3],
                    'scale': 100}}

    return data, label


def get_seg_setting_v2(with_3d=False,
                       pre_render=False,
                       method='',
                       ignore_labels=0,
                       gt_type='',
                       obj_ids=None,
                       label_mapping=None):

    data= OrderedDict([])
    data['data'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}

    if with_3d:
        if pre_render:
            data['label_db'] = {'size': [512, 608],
                    'reader': trans_reader_pre,
                    'reader_params': {'is_gt':True} if method=='gt' \
                            else {},
                    'channel': 1,
                    'is_img': False,
                    'resize_method': cv2.INTER_NEAREST,
                    'transform':label_db_transform,
                    'transform_params':{'with_channel':False}}
        else:
            data['label_db'] = {'size': [512, 608],
                    'reader': trans_reader,
                    'reader_params': {'multi_return': False,
                                      'proj_mat': True},
                    'channel': 1,
                    'is_img': False,
                    'resize_method': cv2.INTER_NEAREST,
                    'transform':label_db_transform,
                    'transform_params':{'with_channel':False,
                        'ignore_labels':ignore_labels}}

    label = OrderedDict([])
    label['softmax_label'] = {'size': [512, 608],
            'channel': 1,
            'resize_method': cv2.INTER_NEAREST,
            'transform':label_transform,
            'transform_params':{
                'ignore_labels':ignore_labels,
                'gt_type':gt_type,
                'obj_ids':obj_ids,
                'label_mapping':label_mapping}}

    return data, label


def get_video_setting(recur_steps, with_db=True):
    data= OrderedDict([])
    label = OrderedDict([])

    for t_i in range(recur_steps):
        data_name = 'image' + str(t_i)
        data[data_name] = {'size': [512, 608],
                    'is_img': True,
                    'resize_method': cv2.INTER_CUBIC,
                    'transform':image_transform,
                    'transform_params':{}}

    if with_db:
        for t_i in range(recur_steps):
            data['label_db' + str(t_i)] = {'size': [512, 608],
                    'is_img': False,
                    'resize_method': cv2.INTER_NEAREST,
                    'transform':label_db_transform,
                    'transform_params':{}}

    for t_i in range(recur_steps):
        if t_i > 0:
            label_name = 'mae_' + str(t_i) + '_label'
            label[label_name] = {'size': [128, 152],
                             'reader': flow_reader,
                             'resize_method': cv2.INTER_NEAREST,
                             'transform': flow_transform,
                             'transform_params':{}}

        label_name = 'softmax_' + str(t_i) + '_label'
        label[label_name] = {'size': [128, 152],
                'resize_method': cv2.INTER_NEAREST,
                'transform':label_transform,
                'transform_params':{}}

    return data, label


def get_video_infer_setting():
    data= OrderedDict([])
    data['image'] = {'size': [512, 608],
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}
    data['image_last'] = {'size': [512, 608],
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}
    data['label_db'] = {'size': [512, 608],
            'resize_method': cv2.INTER_NEAREST,
            'transform':label_db_transform,
            'transform_params':{}}
    data['score_last'] = {'size': [128, 152],
            'resize_method': cv2.INTER_NEAREST,
            'transform':score_transform,
            'transform_params':{}}

    return data


def get_flow_setting():
    data= {}
    data['image1'] = {'size': [512, 608],
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}
    data['image2'] = {'size': [512, 608],
            'resize_method': cv2.INTER_CUBIC,
            'transform': image_transform,
            'transform_params':{}}

    label = {}
    label['mae_label'] = {'size': [128, 152],
                         'reader': flow_reader,
                         'resize_method': cv2.INTER_NEAREST,
                         'transform': flow_transform,
                         'transform_params':{}}
    return data, label


def get_flow_for_seg_setting(has_segment=False):
    data= OrderedDict([])
    data['image'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}

    data['label_db'] = {'size': [512, 608],
            'channel': 1,
            'is_img': False,
            'resize_method': cv2.INTER_NEAREST,
            'transform':label_db_transform,
            'transform_params':{'with_channel':False}}

    label = OrderedDict([])
    label['flow'] = {'size': [512, 608],
            'channel': 2}
    label['mask'] = {'size': [512, 608],
            'channel': 1}

    # whether add segment with spatical transform in
    if has_segment:
        label['segment'] = {'size': [512, 608],
                'channel': 1}

    return data, label


def get_rnn_pose_for_seg_setting(label_path=None,
                                 ignore_labels=None,
                                 label_mapping=None):
    data= OrderedDict([])
    data['image'] = {'size': [512, 608],
            'channel': 3,
            'is_img': True,
            'resize_method': cv2.INTER_CUBIC,
            'transform':image_transform,
            'transform_params':{}}

    data['label_db'] = {'size': [512, 608],
            'channel': 1,
            'is_img': False,
            'resize_method': cv2.INTER_NEAREST,
            'transform':label_db_transform,
            'transform_params':{'with_channel':False}}

    label = OrderedDict([])
    label['segment'] = {'size': [512, 608],
                        'label_path': label_path,
                        'channel': 1,
                        'resize_method': cv2.INTER_NEAREST,
                        'transform': label_transform,
                        'transform_params': {
                                    'ignore_labels': ignore_labels,
                                    'label_mapping': label_mapping}}

    return data, label


def render_from_3d(pose_in, proj, intr, color_map,
        is_color, label_map):
    """
    Inputs:
        pose_in: a input pose with [n x 6] 6 dim vector,
            first 3 is trans and last 3 is rot
    return:
        a list of rendered resutls
    """

    label_proj = []
    sz = proj.get_image_size()

    for pose in pose_in:
        ext = angle_to_proj_mat(
                pose[:3], pose[3:], is_quater=False)
        label_proj_c = proj.pyRenderToMat(intr, ext)
        label_proj.append(
                color_to_id(label_proj_c, sz[0], sz[1],
                color_map,
                is_id=is_color,
                label_mapping=label_map))

    return label_proj



class FlowForSegDataIter(DataIter):
    """Data Iter specified for our rnn flow

    Parameters
    ----------
    """
    def __init__(self,
                 params,
                 root_dir,
                 split,
                 is_augment=True,
                 is_rnn=False,
                 with_flow=False,
                 proj=None,
                 intr_for_render=None,
                 posecnn='',
                 posernn=None,
                 gt_type='bkg',
                 data_setting = None,
                 label_setting = None):
        """
        Input:
            posecnn: posecnn model for loading pre-rendered results
        """

        super(FlowForSegDataIter, self).__init__()
        self.root_dir = root_dir
        self.data_name = data_setting.keys()
        self.data_setting = data_setting
        self.data_num = len(data_setting.keys())
        self._is_augment = is_augment
        self.params = params
        self.split = split
        self.is_rnn = is_rnn
        self._with_flow = with_flow
        self.gt_type = gt_type

        self.res_path = params['pose_rect'] + posecnn + '/'
        if is_rnn:
            _, self.arg_params, self.aux_params = nuts.args_from_models(
                    posernn, False)

        self.proj = proj
        self.intr_for_render = intr_for_render
        key = self.data_setting.keys()[0]
        if 'size' in self.data_setting[key]:
            self.g_shape = self.data_setting[key]['size']

        self.batch_size = params['batch_size']
        self.is_random = True

        # generate sequence of images
        image_list = [line.strip() for line in open(params[self.split + '_set'])]
        cameras = params['cam_names'] if split == 'train' else params['cam_names'][0]

        self.image_seq = image_set_2_seqs(image_list, params['cam_names'])
        if self.split == 'test':
            self.image_seq = self.image_seq[:-1]

        logging.info("generated seq %d \n" % (len(self.image_seq)))
        # self.image_seq = [self.image_seq[i] for i in [0, 1]]

        self.image_list_all = [image_name \
                for image_list in self.image_seq \
                for image_name in image_list]

        if is_augment:
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            self.seq_img = iaa.Sequential([
                sometimes(iaa.Multiply((0.7, 1.2), per_channel=True)),
                # sometimes(iaa.OneOf([
                #     iaa.GaussianBlur((0, 3.0)),
                #     iaa.AverageBlur(k=(2, 7)),
                #     iaa.MedianBlur(k=(3, 11)),
                # ])),
                sometimes(iaa.Grayscale(alpha=(0.0, 1.0)))
                ])

        self.label_name = label_setting.keys()
        self.label_setting = label_setting
        self.label_num = len(self.label_name)
        self.num_data = len(self.image_list_all)
        self.reset()
        self.data, self.label = self._read(0)


    def _read(self, curpose):
        """get two list, each list contains two elements: name and nd.array value"""
        image_file = self.image_list_all[curpose]
        pose = self.pose_seq[curpose]
        data, label= self._read_img(image_file, pose, curpose)
        return list(data.items()), list(label.items())


    def _pose_to_flow(self, image_file, pose, pose_gt, sz, label_db):
        depth_file = self.params['depth_path'] + image_file + '.png'
        depth = self.params['read_depth'](depth_file)
        depth = cv2.resize(depth, (sz[1], sz[0]),
                           interpolation=cv2.INTER_NEAREST)
        order = [3,4,5,0,1,2]
        flow, _, mask = uts_3d.depth2flow(depth, pose_gt[order],
                pose[order], self.params['intrinsic'], is_norm=True)

        return flow, mask


    def _read_img(self, image_file, pose, curpose):
        data_dict = OrderedDict({})
        if self._is_augment:
            seq_img = self.seq_img.to_deterministic()

        intr_key = get_intr_key(self.params['intrinsic'].keys(),
                image_file)

        for data_name in ['image', 'label_db']:
            height, width = self.data_setting[data_name]['size']
            if data_name == 'image':
                image_path = os.path.join(self.params['image_path'],
                        image_file + '.jpg')
                data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            else:
                if self.is_rnn:
                    ext = angle_to_proj_mat(pose[:3], pose[3:])
                    height, width = self.proj.get_image_size()
                    intr_for_render = get_proj_intr(
                            self.params['intrinsic'][intr_key],
                            height, width)
                    label_db = self.proj.pyRenderToMat(
                            intr_for_render, ext)
                    data = color_to_id(label_db, height, width,
                            self.params['color_map'],
                            is_id=self.params['is_color_render'],
                            label_mapping=self.params['id_2_trainid'])
                else:
                    trans_file = self.res_path + image_file \
                            + '_%04d'%self.rand_id[curpose] + '.png'
                    data = cv2.imread(trans_file, cv2.IMREAD_UNCHANGED)
                    data[data == 0] = 1 # set 0 as sky

            if data is None:
                raise ValueError('can not read data %s' % data_name)

            interpolation = self.data_setting[data_name]['resize_method']
            height, width = self.data_setting[data_name]['size']
            is_img = self.data_setting[data_name]['is_img']

            if self._is_augment and is_img:
                data = exp_to_4d(data, self.g_shape, interpolation)
                data = seq_img.augment_images(data)
                data = np.squeeze(data[0])

            data = np.array(data, dtype=np.float32)
            data = cv2.resize(data, (width, height),
                       interpolation=interpolation)

            transform = self.data_setting[data_name]['transform']
            transform_params = self.data_setting[data_name]['transform_params']
            data_dict[data_name] = transform(data, **transform_params)

        # set up label
        label_dict = OrderedDict({})
        if self._with_flow:
            pose_file = self.params['pose_path'] + image_file + '.txt'
            pose_gt = np.loadtxt(pose_file)

            sz = self.label_setting['flow']['size']
            label_id = np.squeeze(data_dict['label_db'])
            flow, mask = self._pose_to_flow(image_file, pose,
                    pose_gt, sz,
                    label_id)
            label_dict['flow'] = flow_transform(flow)
            label_dict['mask'] = label_db_transform(mask)

        for label_name in self.label_setting.keys():
            sz = self.label_setting[label_name]['size']

            label_path = self.label_setting[label_name]['label_path'] + \
                        image_file + '.png'
            label_gt = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            label_gt = cv2.resize(label_gt, (sz[1], sz[0]),
                    interpolation=cv2.INTER_NEAREST)
            transform = self.label_setting[label_name]['transform']
            transform_params = self.label_setting[label_name]['transform_params']
            label_dict['segment'] = transform(label_gt, **transform_params)

        return data_dict, label_dict

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.label]

    def get_batch_size(self):
        return self.batch_size


    def reset_rnn(self):
        # sample pose into a sequence
        self.pose_seq = []

        for count, image_list in enumerate(self.image_seq):
            logging.info('predict sequence %d len: %d' % (
                count, len(image_list)))
            image_num = len(image_list)
            data_names = ['pose_in_%03d'%i for i in range(image_num)]
            data_shapes = [tuple([1, 6]) for i in range(image_num)]
            data_shapes = [(name, shape) for name, shape \
                    in zip(data_names, data_shapes)]

            inputs = nuts.get_mx_var_by_name(data_names)
            # only accept hier rnn & layer number = 2
            net = posenet.recurrent_pose(inputs, name='pose',
                    is_train=False, is_highorder=True, layer_num=2)
            net = mx.sym.Group(net.values())
            mod = mx.mod.Module(net,
                                data_names=data_names,
                                label_names=None,
                                context=mx.gpu(1))
            mod.bind(for_training=False, data_shapes=data_shapes)
            mod.set_params(self.arg_params,
                           self.aux_params,
                           allow_missing=False)

            pose_in = []
            for count, image_name in enumerate(image_list):
                trans_file = self.res_path + image_name + '.txt'
                cur_pose = trans_reader_pre(trans_file, ext='txt',
                        max_pre_render_num=self.params['pose_permute_num'])
                cur_pose = pose_transform(cur_pose)
                pose_in.append(mx.nd.array(cur_pose))

            mod.forward(Batch(pose_in))
            output_nd = mod.get_outputs()
            for res in output_nd:
                self.pose_seq.append(res.asnumpy()[0])


    def reset_posenet(self):
        self.pose_seq = []
        self.rand_id = []
        for image_list in self.image_seq:
            for count, image_name in enumerate(image_list):
                trans_file = self.res_path + image_name + '.txt'
                cur_pose, idx = trans_reader_pre(trans_file, ext='txt',
                        get_rand_id=True,
                        max_pre_render_num=self.params['pose_permute_num'])

                self.pose_seq.append(cur_pose)
                self.rand_id.append(idx)


    def reset(self):
        if self.is_rnn:
            self.reset_rnn()
        else:
            self.reset_posenet()

        logging.info("generate new image seq")
        self.cursor = -1 * self.batch_size
        self.order = np.random.permutation(self.num_data)

    def iter_next(self):
        if(self.cursor < self.num_data-1):
            self.cursor += self.batch_size
            self.cursor = min(self.cursor, self.num_data-1)
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            batch_data = [mx.nd.empty(info[1]) for info in self.provide_data]
            batch_label = [mx.nd.empty(info[1]) for info in self.provide_label]
            end_curpose = min(self.cursor + self.batch_size, self.num_data)
            pose = 0
            for i in xrange(self.cursor, end_curpose):
                self.data, self.label = self._read(self.order[i])
                for info_idx, (key, value) in enumerate(self.data):
                    batch_data[info_idx][pose] = value[0]
                for info_idx, (key, value) in enumerate(self.label):
                    batch_label[info_idx][pose] = value[0]
                pose += 1

            return DataBatch(data=batch_data, label=batch_label,
                    pad=self.batch_size-pose)

        else:
            raise StopIteration



class SegDataIter(DataIter):
    """
    Parameters
    ----------
    """
    def __init__(self,
                 params,
                 root_dir,
                 flist_name,
                 is_augment = True,
                 proj=None,
                 intr_for_render=None,
                 cut_off_size = None,
                 data_setting = None,
                 label_setting = None):
        """
        Inputs:
            proj: the projector

        """

        super(SegDataIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.cut_off_size = cut_off_size
        self.data_name = data_setting.keys()
        self.data_setting = data_setting
        self.data_num = len(data_setting.keys())
        self._is_augment = is_augment
        self.params = params

        self.proj = proj
        self.intr_for_render = intr_for_render
        key = self.data_setting.keys()[0]
        if 'size' in self.data_setting[key]:
            self.g_shape = self.data_setting[key]['size']

        self.batch_size = params['batch_size']
        self.is_random = True

        if is_augment:
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            self.seq_both = iaa.Sequential([
                sometimes(iaa.Crop(percent=(0, 0.2), keep_size=False)),
                iaa.Fliplr(0.5),
                ])

            # augment that only apply to images
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            self.seq_img = iaa.Sequential([
                sometimes(iaa.Multiply((0.7, 1.2), per_channel=True)),
                sometimes(iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 4)),
                    # iaa.MedianBlur(k=(3, 7)),
                ])),
                sometimes(iaa.Grayscale(alpha=(0.0, 1.0)))
                ])

        self.label_name = label_setting.keys()
        self.label_setting = label_setting
        self.label_num = len(self.label_name)

        self.filenames = [line.strip('\n') for line in open(self.flist_name, 'r')]
        self.num_data = len(self.filenames)
        self.data, self.label = self._read(0)

        self.reset()

    def _read(self, curpose):
        """get two list, each list contains two elements: name and nd.array value"""
        self.input_names = self.filenames[curpose].split("\t")
        data, label = self._read_img(self.input_names[:self.data_num],
                self.input_names[self.data_num:])

        return list(data.items()), list(label.items())


    def _read_img(self, data_files, label_files):
        data_dict = OrderedDict({})

        if self._is_augment:
            seq_both = self.seq_both.to_deterministic()
            seq_img = self.seq_img.to_deterministic()

        intr_key = get_intr_key(self.params['intrinsic'].keys(), data_files[0])
        for i, (data_name, filename) in enumerate(zip(self.data_name, data_files)):
            if not ('reader' in self.data_setting[data_name].keys()):
                data = cv2.imread(os.path.join(self.root_dir, filename),
                       cv2.IMREAD_UNCHANGED)
            else:
                args = self.data_setting[data_name]['reader_params']
                data = self.data_setting[data_name]['reader'](
                       os.path.join(self.root_dir, filename), **args)
                if self.proj and data_name == 'label_db':
                    label_db = self.proj.pyRenderToMat(
                            self.params['intrinsic'][intr_key], data)
                    height, width = self.data_setting['data']['size']
                    data = color_to_id(
                            label_db,
                            height, width,
                            color_map=self.params['color_map'],
                            is_id=self.params['is_color_render'],
                            label_mapping=self.params['id_2_trainid'])

            is_map = 'resize_method' in self.data_setting[data_name]
            if is_map:
                interpolation = self.data_setting[data_name]['resize_method']
                height, width = self.data_setting[data_name]['size']

                if self._is_augment:
                    data = exp_to_4d(data, self.g_shape, interpolation)
                    data = seq_both.augment_images(data)
                    if self.data_setting[data_name]['is_img']:
                        data = seq_img.augment_images(data)
                    data = np.squeeze(data[0])

                data = np.array(data, dtype=np.float32)
                data = cv2.resize(data, (width, height),
                           interpolation=interpolation)

            transform = self.data_setting[data_name]['transform']
            transform_params = self.data_setting[data_name]['transform_params']
            data_dict[data_name] = transform(data, **transform_params)

        label_dict = OrderedDict({})
        for i, (label_name, filename) in enumerate(zip(self.label_name, label_files)):
            if not ('reader' in self.label_setting[label_name].keys()):
                label = cv2.imread(os.path.join(self.root_dir, filename),
                        cv2.IMREAD_UNCHANGED)
            else:
                args = self.label_setting[label_name]['reader_params']
                label = self.label_setting[label_name]['reader'](
                        os.path.join(self.root_dir, filename), **args)

            is_map = 'resize_method' in self.label_setting[label_name]
            if is_map:
                interpolation = self.label_setting[label_name]['resize_method']
                height, width = self.label_setting[label_name]['size']

                if self._is_augment:
                    label = exp_to_4d(label, self.g_shape, interpolation)
                    label = seq_both.augment_images(label)
                    label = np.squeeze(label[0]) # reading batch size = 1

                label = np.array(label, dtype=np.float32)
                label = cv2.resize(label, (width, height),
                                   interpolation=interpolation)

            transform = self.label_setting[label_name]['transform']
            trans_params = self.label_setting[label_name]['transform_params']

            label_dict[label_name] = transform(label, **trans_params)
            # print np.unique(label_dict[label_name])

        return data_dict, label_dict

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(key, tuple([self.batch_size] + list(value.shape[1:])))
                for key, value in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.cursor = -1 * self.batch_size
        self.order = np.random.permutation(self.num_data)

    def iter_next(self):
        if(self.cursor < self.num_data-1):
            self.cursor += self.batch_size
            self.cursor = min(self.cursor, self.num_data-1)
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            batch_data = [mx.nd.empty(info[1]) for info in self.provide_data]
            batch_label = [mx.nd.empty(info[1]) for info in self.provide_label]
            end_curpose = min(self.cursor + self.batch_size, self.num_data)
            pose = 0
            for i in xrange(self.cursor, end_curpose):
                self.data, self.label = self._read(self.order[i])
                for info_idx, (key, value) in enumerate(self.data):
                    batch_data[info_idx][pose] = value[0]
                for info_idx, (key, value) in enumerate(self.label):
                    batch_label[info_idx][pose] = value[0]
                pose += 1

            return DataBatch(data=batch_data, label=batch_label,
                    pad=self.batch_size-pose)

        else:
            raise StopIteration


class PoseDataIter(DataIter):
    def __init__(self,
                 params,
                 root_dir,
                 flist_name,
                 is_augment=True,
                 proj=None,
                 intr_for_render=None,
                 sem_weight=False,
                 data_setting = None,
                 label_setting = None):

        super(PoseDataIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.data_name = data_setting.keys()
        self.data_setting = data_setting
        self.data_num = len(data_setting.keys())
        self._is_augment = is_augment
        self.params = params
        self.proj = proj
        self.intr_for_render = intr_for_render
        self.sem_weight=sem_weight

        if 'image' in self.data_setting:
            self.g_shape = self.data_setting['image']['size']

        if is_augment:
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            # augment that only apply to images to avoid overfitting
            self.seq_img = iaa.Sequential([
                sometimes(iaa.Multiply((0.7, 1.2), per_channel=True)),
                sometimes(iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 5)),
                    # iaa.MedianBlur(k=(3, 7)),
                ])),
                sometimes(iaa.Grayscale(alpha=(0.0, 1.0)))
                ])

        self.label_name = label_setting.keys()
        self.label_setting = label_setting
        self.label_num = len(self.label_name)

        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.filenames = [line.strip() for line in open(self.flist_name)]
        self.data, self.label = self._read(0)
        self.reset()


    def _read(self, curpose):
        """get two list, each list contains two elements: name and nd.array value"""
        self.input_names = self.filenames[curpose].split('\t')
        data, label= self._read_img(self.input_names[:self.data_num],
                self.input_names[self.data_num:])

        return list(data.items()), list(label.items())



    def _read_img(self, data_files, label_files):
        data_dict = OrderedDict({})
        label_dict = OrderedDict({})

        if self._is_augment:
            seq_img = self.seq_img.to_deterministic()

        intr_key = get_intr_key(self.params['intrinsic'].keys(), data_files[0])
        for i, (data_name, filename) in enumerate(zip(self.data_name, data_files)):
            if not ('reader' in self.data_setting[data_name].keys()):
                data = cv2.imread(os.path.join(self.root_dir, filename),
                       cv2.IMREAD_UNCHANGED)
            else:
                args = self.data_setting[data_name]['reader_params']
                if data_name == 'points':
                    args.update({'K' : self.params['intrinsic'][intr_key],
                                 'sz': self.data_setting[data_name]['size']})

                    if self.sem_weight:
                        image_name = '/'.join(filename.split('/')[1:])
                        new_args = {'label_file': self.params['label_bkgfull_path'] + image_name,
                                    'label_weight': self.params['label_weight']}
                        args.update(new_args)

                    data, mask = self.data_setting[data_name]['reader'](
                           os.path.join(self.root_dir, filename), **args)
                else:
                    data = self.data_setting[data_name]['reader'](
                           os.path.join(self.root_dir, filename), **args)

                if data_name == 'label_db' or data_name == 'pose_in':
                    pre_render = (self.data_setting[data_name]['reader'] == trans_reader_pre_all)
                    if pre_render:
                        motion_perturb, data = data
                        data[data > 200] = 0
                    else:
                        motion, motion_perturb, motion_mat, motion_mat_gt = data
                        data = motion_perturb
                        if self.proj:
                            height, width = self.g_shape
                            intr_for_render = get_proj_intr(
                                    self.params['intrinsic'][intr_key], height, width)
                            label_db = self.proj.pyRenderToMat(intr_for_render, motion_mat)
                            data = color_to_id(
                                    label_db,
                                    height, width,
                                    color_map=self.params['color_map'],
                                    is_id=self.params['is_color_render'],
                                    label_mapping=self.params['id_2_trainid'])

            is_map = 'resize_method' in self.data_setting[data_name]
            if is_map:
                interpolation = self.data_setting[data_name]['resize_method']
                height, width = self.data_setting[data_name]['size']

                if self._is_augment and self.data_setting[data_name]['is_img']:
                    data = exp_to_4d(data, self.g_shape, interpolation)
                    data = seq_img.augment_images(data)
                    data = np.squeeze(data[0])

                data = np.array(data, dtype=np.float32)
                data = cv2.resize(data, (width, height),
                           interpolation=interpolation)

            transform = self.data_setting[data_name]['transform']
            trans_params = self.data_setting[data_name]['transform_params']
            data_dict[data_name] = transform(data, **trans_params)

            # directly put in input label and gt
            if data_name == 'points':
                data_dict['weight'] = np.expand_dims(mask, axis=0)  # (1, point_num)

            if data_name == 'label_db':
                data_dict['pose_in'] = pose_transform(motion_perturb)
                if not pre_render:
                    label_dict['pose'] = pose_transform(motion)

        for i, (label_name, filename) in enumerate(zip(self.label_name, label_files)):
            if not ('reader' in self.label_setting[label_name].keys()):
                label = cv2.imread(os.path.join(self.root_dir, filename),
                                   cv2.IMREAD_UNCHANGED)
            else:
                args = self.label_setting[label_name]['reader_params']
                label = self.label_setting[label_name]['reader'](
                        os.path.join(self.root_dir, filename), **args)

                if label_name == 'pose':
                    if self.label_setting[label_name]['reader'] == trans_reader:
                        motion, motion_purturb, motion_mat, motion_mat_gt = label
                        label = motion

            is_map = 'resize_method' in self.label_setting[label_name]
            if is_map:
                interpolation = self.label_setting[label_name]['resize_method']
                height, width = self.label_setting[label_name]['size']
                label = np.array(label, dtype=np.float32)
                label = cv2.resize(label, (width, height),
                                   interpolation=interpolation)

            transform = self.label_setting[label_name]['transform']
            trans_params = self.label_setting[label_name]['transform_params']
            label_dict[label_name] = transform(label, **trans_params)

        return data_dict, label_dict

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(key, tuple([1] + list(value.shape[1:])))
                for key, value in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(key, tuple([1] + list(value.shape[1:])))
                for key, value in self.label]

    def get_batch_size(self):
        return 1

    def reset(self):
        self.cursor = -1
        self.order = range(self.num_data)
        # self.order = np.random.permutation(self.num_data)

    def iter_next(self):
        if(self.cursor < self.num_data - 1):
            self.cursor += 1
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            # pdb.set_trace()
            self.data, self.label = self._read(self.order[self.cursor])
            data = [mx.nd.array(data[1]) for data in self.data]
            label = [mx.nd.array(label[1]) for label in self.label]
            return DataBatch(data=data, label=label)

        else:
            raise StopIteration


class FlowDataIter(SegDataIter):
    """Data reader for image flow
    """
    def __init__(self,
                 params,
                 root_dir,
                 flist_name,
                 iter_num=0,
                 iter_suffix=None,
                 cut_off_size=None,
                 data_setting=None,
                 label_setting=None):

        super(FlowDataIter, self).__init__(params,
                     root_dir,
                     flist_name,
                     cut_off_size,
                     data_setting,
                     label_setting)
        self.iter_suffix = iter_suffix
        self.iter_num = iter_num


    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        data_all = [(key, tuple([1] + list(value.shape[1:])))
                    for key, value in self.data]
        data_all.append(('mask',
            tuple([1, 1] + list(self.label[0][1].shape[2:]))))

        return data_all


    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        label_all = []
        for key, value in self.label:
            # copy label based on iter and scale
            loss_name = key.split('_')[0]
            for suffix in self.iter_suffix:
                if self.iter_num > 0:
                    for i in range(self.iter_num):
                        key_name = loss_name + suffix + '_' + str(i) + '_label'
                        label_all.append(
                                (key_name, tuple([1] + list(value.shape[1:]))))
                else:
                    key_name = loss_name + suffix + '_label'
                    label_all.append(
                            (key_name, tuple([1] + list(value.shape[1:]))))

        return label_all


    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            data = [mx.nd.array(data[1]) for data in self.data]
            # generate a mask for flow learning
            for key, value in self.label:
                mask = np.float32(np.sum(np.abs(value), axis=1) > 0)
                mask = np.expand_dims(mask, axis=0)  # (1, c, h, w)
            data.append(mx.nd.array(mask))

            label = []
            for key, value in self.label:
                # copy label based on iter and scale
                for suffix in self.iter_suffix:
                    if self.iter_num > 0:
                        for i in range(self.iter_num):
                            label.append(mx.nd.array(value))
                    else:
                        label.append(mx.nd.array(value))

            return DataBatch(data=data, label=label)

        else:
            raise StopIteration


class VideoSegDataIter(SegDataIter):
    def __init__(self,
                 params,
                 root_dir,
                 flist_name,
                 spatial_iter_num,
                 temporal_iter_num,
                 loss_suffix=[''],
                 data_setting=None,
                 label_setting=None):
        # pdb.set_trace()
        super(VideoSegDataIter, self).__init__(params,
                     root_dir,
                     flist_name,
                     data_setting=data_setting,
                     label_setting=label_setting)
        self.loss_suffix = loss_suffix
        self.s_num = spatial_iter_num
        self.t_num = temporal_iter_num


    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        label_all = []
        for key, value in self.label:
            # copy label based on iter and scale
            label_info = key.split('_')
            loss_name = label_info[0]
            time_str = label_info[1]
            assert loss_name in ['softmax', 'mae']
            i = 0 if loss_name == 'mae' else 1
            for suffix in self.loss_suffix[i]:
                for s_i in range(self.s_num[i]):
                    key_name = loss_name + suffix + '_' + \
                               str(s_i) + '_' + time_str + '_label'
                    label_all.append(
                       (key_name, tuple([1] + list(value.shape[1:]))))

        return label_all


    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            data = [mx.nd.array(data[1]) for data in self.data]
            label = []
            for key, value in self.label:
                # copy label based on iter and scale
                label_info = key.split('_')
                loss_name = label_info[0]
                i = 0 if loss_name == 'mae' else 1
                for suffix in self.loss_suffix[i]:
                    for s_i in range(self.s_num[i]):
                        label.append(mx.nd.array(value))

            return DataBatch(data=data, label=label)

        else:
            raise StopIteration


def merge_labels(label_bkg, label_obj, label_sky,
                 color_map, color_map_list, building_id,
                 sky_id):
    """Merge different labels to a single semantic label
    """
    import cython_util as cuts

    label_sky = uts.color2label(label_sky, color_map)
    label_full = np.zeros_like(label_sky)
    label_full[label_sky == sky_id] = sky_id

    label_bkg = uts.color2label(label_bkg, color_map)
    label_full[label_bkg > 0] = label_bkg[label_bkg > 0]
    label_full = cuts.extend_building(np.int32(label_full),
                                      building_id,
                                      sky_id)
    label_obj = uts.color2label(label_obj, color_map)

    label_full[label_obj > 0] = label_obj[label_obj > 0]
    label_full[label_full > len(color_map_list)] = 0
    invalid_color = [255, 255, 255]
    label_color = uts.label2color(label_full, color_map_list,
            bkg_color=invalid_color)

    return label_color[:, :, ::-1], label_full


