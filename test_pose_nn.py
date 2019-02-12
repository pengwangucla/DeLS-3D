"""Test demo for trained model
"""
import os
import argparse
import logging
import cv2
import mxnet as mx
import numpy as np
import metric
import time
import vis_utils.utils as uts
import vis_utils.utils_3d as uts_3d


import dataset.zpark as zpark
import dataset.dlake as dlake
data_libs = {}
data_libs['zpark'] = zpark
data_libs['dlake'] = dlake

import dataset.data_iters as data_iter
import networks.pose_nn as pose_nn
import networks.net_util as nuts
from collections import namedtuple, OrderedDict
import pdb


np.set_printoptions(precision=4, suppress=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
Batch = namedtuple('Batch', ['data'])


def softmax(x, axis):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def to_prob(x, axis):
    num = x.shape[axis]
    temp_sum = np.expand_dims(np.sum(x, axis=axis),
                              axis=axis)
    return x / np.repeat(temp_sum, num, axis=axis)


def infer(mod, inputs,
          data_setting,
          out_names):

    assert isinstance(inputs, OrderedDict)
    data_list = []
    s = time.time()
    for in_name in inputs:
        data = np.array(inputs[in_name], dtype=np.float32)  # (h, w, c)
        if 'size' in data_setting[in_name]:
            height, width = data_setting[in_name]['size']
            interpolation = data_setting[in_name]['resize_method']
            data = cv2.resize(data, (width, height),
                    interpolation=interpolation)
        transform = data_setting[in_name]['transform']
        args = data_setting[in_name]['transform_params']
        data = transform(data, **args)
        data_list.append(mx.nd.array(data))

    prep_time = time.time() - s
    mod.forward(Batch(data_list))
    output_nd = mod.get_outputs()

    res = {}
    for i, name in enumerate(out_names):
        output_nd[i].wait_to_read()
        asnp_time = time.time() - prep_time - s
        res[name] = output_nd[i].asnumpy()

    post_time = time.time() - prep_time - s- asnp_time
    logging.info('forward time pre {}, asnp {}, post {}'.format(
        prep_time, asnp_time, post_time))

    return res


def iter_infer(mod, data_batch, out_names):
    s = time.time()
    mod.forward(data_batch)
    output_nd = mod.get_outputs()

    res = OrderedDict({})
    for i, name in enumerate(out_names):
        output_nd[i].wait_to_read()
        res[name] = output_nd[i].asnumpy()
    asnp_time = time.time() - s
    logging.info('forward time {}'.format(asnp_time))
    return res


def def_model(in_names, params=None, version='v2', args=None):
    """ set up the symbolic for data
    """

    inputs = nuts.get_mx_var_by_name(in_names)
    if version == 'v1':
        net = pose_nn.pose_block(inputs,
                    params,
                    in_names=in_names,
                    name='pose',
                    is_train=False)

    elif version == 'v2':
        net = pose_nn.pose_nn(inputs[in_names[0]], name='pose',
                pose_in=inputs[in_names[1]], is_train=False)

    elif version == 'rnn':
        net = pose_nn.recurrent_pose(inputs, name='pose',
                is_train=False, **args)

    else:
        raise ValueError('no such network')

    return mx.symbol.Group(net.values()), net.keys()


def posenet_res2label(pose_output, proj, intrinsic, sz, color_map):
    ext_pred = data_iter.angle_to_proj_mat(
            pose_output[0, :3], pose_output[0, 3:], is_quater=False)
    label_proj_c = proj.pyRenderToMat(intrinsic, ext_pred)
    label_proj = data_iter.color_to_id(
            label_proj_c, sz[0], sz[1], color_map)
    return label_proj


def get_input_shape(data_config, name):
    if 'center_crop' in data_config[name]['transform_params']:
        crop=data_config[name]['transform_params']['center_crop']
        return [crop, crop]
    return data_config[name]['size']


def test_posenet_v1(method, dataname, is_eval=True):
    ctx = mx.gpu(int(args.gpu_ids))
    params = eval('data.' + dataname + '.set_params()')

    data_setting, label_setting = data_iter.get_pose_setting(
            pre_render=False)
    height, width = data_setting['image']['size']
    proj = data_iter.get_projector(params, height, width)
    val_iter = data_iter.PoseDataIter(params=params,
            root_dir=params['data_path'],
            flist_name='train_file/test_pose.ls',
            is_augment=False,
            proj=proj,
            data_setting=data_setting,
            label_setting=label_setting)

    # define networks
    height, width = data_setting['image']['size']
    data_names = ['image', 'label_db', 'pose_in']
    data_shapes = OrderedDict(val_iter.provide_data)
    data_shapes = [data_shapes[name] for name in data_names]

    net, out_names = def_model(data_names, params, version='v1')
    in_model = {'model':args.model_path + method}
    model = nuts.load_model(in_model,
                data_names=data_names,
                data_shapes=data_shapes,
                net=net, ctx=ctx)

    # start eval
    pose_eval_metric = metric.PoseMetric(is_euler=True)
    seg_eval_metric = metric.SegMetric(ignore_label=255)

    while val_iter.iter_next():
        # obtain an instance of output
        path = val_iter.input_names[0].split('/')
        image_name = path[-1][:-4]
        test_scene = '/'.join(path[1:-1])

        logging.info('results for {}, {}'.format(test_scene, image_name))
        save_path = params['output_path'] + method + '/' + test_scene + '/'
        uts.mkdir_if_need(save_path)

        data = [mx.nd.array(data[1]) for data in val_iter.data]
        pose = [mx.nd.array(label[1]) for label in val_iter.label]

        # get pose and corresponding projected map
        res = iter_infer(model, Batch(data), out_names)
        pose_output = res[out_names[0]]

        intr_key = data_iter.get_intr_key(params['intrinsic'].keys(), path)
        intr = data_iter.get_proj_intr(params['intrinsic'][intr_key],
                height, width)
        label_proj = data_iter.render_from_3d(pose_output,
                             proj, intr, params['color_map'],
                             is_color=params['is_color_render'],
                             label_map=params['id_2_trainid'])[0]

        if is_eval:
            motion_gt = pose[0].asnumpy()[0]
            motion_noisy = data[2].asnumpy()[0]
            label_in = data[1].asnumpy()[0, 0]
            label_gt = cv2.imread(params['label_bkg_path'] + test_scene + '/' + image_name + '.png', cv2.IMREAD_UNCHANGED)
            label_seg = mx.nd.array(data_iter.label_transform(label_gt))

            # pose metric & segment metric
            seg_output = mx.nd.array(label_proj[None, :, :])
            seg_output = mx.ndarray.one_hot(seg_output, params['class_num'])
            seg_output = mx.ndarray.transpose(seg_output, axes=(0, 3, 1, 2))

            pose_eval_metric.update([mx.nd.array(motion_gt)],
                    [mx.nd.array(pose_output)])
            seg_eval_metric.update(label_seg, [seg_output])
            logging.info('Eval seg {}'.format(seg_eval_metric.get()))
            logging.info('Eval pose {}'.format(pose_eval_metric.get()))

        if args.is_save:
            logging.info('Saved')
            cv2.imwrite(save_path + image_name + '_label_perturb.png', label_in)
            np.savetxt(save_path + image_name + '_perturb.txt', motion_noisy)
            cv2.imwrite(save_path + image_name + '_label_rect.png', label_proj)
            np.savetxt(save_path + image_name + '_rect.txt', pose_output[0, :])
        val_iter.data, val_iter.label = val_iter._read()

    logging.info('Final seg eval {} pose eval {}'.format(str(seg_eval_metric.get()), str(pose_eval_metric.get())))


def test_posenet_v1_demo(method, dataname='dlake'):
    logging.info(method)

    ctx = mx.gpu(int(args.gpu_ids))
    params = eval('data.' + dataname + '.set_params()')

    # define saving place
    data_setting, label_setting = data_iter.get_pose_setting(
             with_pose_in=True)
    height, width = data_setting['image']['size']
    proj = data_iter.get_projector(params, height, width)

    # init model
    data_names = ['image', 'label_db', 'pose_in']
    data_shapes = [tuple([1, 3] + data_setting['image']['size']),
                   tuple([1, 1] + data_setting['label_db']['size']),
                   tuple([1, 6])]

    net, out_names = def_model(data_names, params, version='v1')
    in_model = {'model': args.model_path + method}
    model = nuts.load_model(in_model, data_names=data_names,
                   data_shapes=data_shapes, net=net, ctx=ctx)

    image_list = [line.strip() for line in open(params['test_set'])]

    # start eval
    pose_eval_metric = metric.PoseMetric(is_euler=True)
    seg_eval_metric = metric.SegMetric(ignore_label=255)

    image_num = len(image_list)
    for i, image_name in enumerate(image_list):
        logging.info('%d/%d, %s'%(i, image_num, image_name))
        intr_key = data_iter.get_intr_key(
                params['intrinsic'].keys(), image_name)
        intr = data_iter.get_proj_intr(
                params['intrinsic'][intr_key],
                height, width)

        # get input
        pose_gt = np.loadtxt(params['pose_path'] + image_name + '.txt')
        trans_i, rot_i = uts_3d.random_perturb(
                           pose_gt[:3], pose_gt[3:], 5.0)
        pose_in = np.concatenate([trans_i, rot_i])
        # pose_in = np.loadtxt(params['pose_permute'] + image_name + \
        #                      "_%04d.txt" % 0)
        label_db = data_iter.render_from_3d(pose_in[None, :],
                                     proj, intr, params['color_map'],
                                     is_color=params['is_color_render'],
                                     label_map=params['id_2_trainid'])[0]

        image = cv2.imread(params['image_path'] + image_name + '.jpg')
        inputs = OrderedDict([('image', image),
                              ('label_db', label_db),
                              ('pose_in', pose_in)])

        # get pose and corresponding projected map
        res = infer(model, inputs, data_setting, out_names)
        pose_out = res[out_names[0]]

        label_res = data_iter.render_from_3d(pose_out , proj,
                intr, params['color_map'],
                is_color=params['is_color_render'],
                label_map=params['id_2_trainid'])[0]

        # evaluation pose metric & segment metric
        label_gt = cv2.imread(params['label_bkg_path'] + image_name + '.png',
                              cv2.IMREAD_UNCHANGED)
        label_seg = mx.nd.array(data_iter.label_transform(label_gt))

        seg_output = mx.nd.array(label_res[None, :, :])
        seg_output = mx.ndarray.one_hot(seg_output, params['class_num'])
        seg_output = mx.ndarray.transpose(seg_output, axes=(0, 3, 1, 2))
        pose_eval_metric.update([mx.nd.array(pose_gt[None, :])],
                                [mx.nd.array(pose_out)])
        seg_eval_metric.update(label_seg, [seg_output])

        logging.info('Eval seg {}'.format(seg_eval_metric.get()))
        logging.info('Eval pose {}'.format(pose_eval_metric.get()))

        if args.is_save:
            test_scene = '/'.join(image_name.split('/')[:-1])
            save_path = params['output_path']+method+'/'+test_scene
            uts.mkdir_if_need(save_path)
            cv2.imwrite(params['output_path'] + method + '/' + \
                    image_name + '.png', label_res)
            np.savetxt(params['output_path'] + method + '/' + \
                    image_name + '.txt', pose_out)

    logging.info('Final seg eval {} pose eval {}'.format(str(seg_eval_metric.get()), str(pose_eval_metric.get())))


def test_posenet(is_eval=True):
    """test a single network performance
    """
    ctx = mx.gpu(int(args.gpu_ids))
    params = zpark.set_params()
    data_config, label_config = data_iter.get_posenet_setting(with_gps=True)
    height, width = data_config['image']['size']
    height *= 2
    width *= 2

    proj, intr_for_render = data_iter.get_projector(params, height, width)
    data_names = ['image', 'pose_in']
    data_shapes = [tuple([1, 3] + get_input_shape(data_config,
        data_names[0]))]
    data_shapes = data_shapes + [tuple([1, 7])]

    net, out_names = def_model(data_names)
    model = nuts.load_model({'model':args.test_model},
            data_names=data_names, data_shapes=data_shapes,
            net=net, ctx=ctx)

    test_scene = params['test_scene'][0]
    image_path = params['image_path'] + test_scene + '/'
    trans_path = params['output_path'] + 'Loc/' + test_scene + '/'
    label_path = params['label_path'] + test_scene + '/'
    method = args.test_model.split('/')[-1]

    image_list = uts.list_images(image_path)
    save_path = params['output_path'] + method + '/' + test_scene + '/'
    uts.mkdir_if_need(save_path)

    pose_eval_metric = metric.PoseMetric()
    seg_eval_metric = metric.SegMetric(ignore_label=0)

    time_all = 0.
    for count, image_name in enumerate(image_list):
        image_name = image_name.split('/')[-1]
        if os.path.exists(save_path + image_name[:-4] + '.png') and False:
            continue

        print '\t test image {}'.format(image_name),
        image = cv2.imread(image_path + '/' + image_name)
        motion_gt, motion_noisy, ext, ext_gt = data_iter.trans_reader(
                trans_path+ '/' + image_name[:-4] + '.txt',
                multi_return=True, convert_to='qu', proj_mat=True)
        label = cv2.imread(label_path + '/' + image_name[:-4] + '.png',
                           cv2.IMREAD_UNCHANGED)

        label_gt_c = proj.pyRenderToMat(intr_for_render, ext_gt)
        label_gt = data_iter.color_to_id(label_gt_c, height, width, params['color_map'])

        label_in_c = proj.pyRenderToMat(intr_for_render, ext)
        label_in = data_iter.color_to_id(label_in_c, height, width, params['color_map'])

        start_time = time.time()
        inputs = {data_names[0]: image, 'pose_in': motion_noisy}

        res = infer(model, inputs, data_config, out_names)
        pose_output = np.concatenate(res.values(), axis=1)
        ext_pred = data_iter.angle_to_proj_mat(pose_output[0, :3],
                pose_output[0, 3:], is_quater=True)

        label_proj_c = proj.pyRenderToMat(intr_for_render, ext_pred)
        label_proj = data_iter.color_to_id(label_proj_c, height, width,
                                        params['color_map'])

        # update metrics
        if is_eval:
            # pose metric & segment metric
            label = mx.nd.array(data_iter.label_transform(label_gt))
            motion = mx.nd.array(data_iter.pose_transform(motion_gt))
            seg_output = mx.nd.array(label_proj[None, :, :])
            seg_output = mx.ndarray.one_hot(seg_output, params['class_num'])
            seg_output = mx.ndarray.transpose(seg_output, axes=(0, 3, 1, 2))
            pose_output_mx = mx.nd.array(pose_output)

            seg_eval_metric.update([label], [seg_output])
            pose_eval_metric.update([motion], [pose_output_mx])
            logging.info('Eval seg {}'.format(seg_eval_metric.get()))
            logging.info('Eval pose {}'.format(pose_eval_metric.get()))

        cost_time = time.time() - start_time
        time_all += cost_time
        print "\t infer time cost {}".format(cost_time)

        if args.is_save:
            is_write = cv2.imwrite(save_path + image_name[:-4] + '_label.png', label_gt)
            is_write = cv2.imwrite(save_path + image_name[:-4] + '_label_perturb.png', label_in)
            np.savetxt(save_path + image_name[:-4] + '_perturb.txt', motion_noisy)
            is_write = cv2.imwrite(save_path + image_name[:-4] + '_label_rect.png', label_proj)
            np.savetxt(save_path + image_name[:-4] + '_rect.txt', pose_output[0, :])
            if not is_write:
                raise ValueError('not able to write')

    logging.info('Final seg eval {} pose eval {} ave time {}'.format(
        str(seg_eval_metric.get()), str(pose_eval_metric.get()),
        time_all / len(image_list)))


def rect_pre_render_res(method, dataname='dlake'):
    # read image & pre render image
    ctx = mx.gpu(int(args.gpu_ids))
    params = eval('data.' + dataname + '.set_params()')

    n_of_permute = 10
    data_config, label_config = data_iter.get_pose_setting(
            with_points=False, with_pose_in=True)
    height, width = data_config['image']['size']

    proj = data_iter.get_projector(params, height, width)
    data_names = ['image', 'label_db', 'pose_in']
    sz = get_input_shape(data_config, data_names[0])
    data_shapes = [tuple([1, 3] + sz)]
    data_shapes = data_shapes + [tuple([1, 1] + sz)]
    data_shapes = data_shapes + [tuple([1, 6])]

    net, out_names = def_model(data_names, params, version='v1')
    in_model = {'model': args.model_path + method}
    model = nuts.load_model(in_model, data_names=data_names,
                  data_shapes=data_shapes, net=net, ctx=ctx)

    permute_path = params['pose_permute']
    save_path = params['pose_rect'] + "%s/" % (method)

    all_images = [line.strip() for line in open(params['train_set'])]
    all_images += [line.strip() for line in open(params['test_set'])]

    # pose_eval_metric = metric.PoseMetric(is_euler=True)

    for count, image_name in enumerate(all_images):
        parts = image_name.split('/')
        test_scene = '/'.join(parts[:-1])
        cur_path = save_path + test_scene
        uts.mkdir_if_need(cur_path)

        intr_key = data_iter.get_intr_key(
                params['intrinsic'].keys(), image_name)
        intr = data_iter.get_proj_intr(
                params['intrinsic'][intr_key], height, width)

        image = cv2.imread(params['image_path'] + image_name + '.jpg')
        for i in range(n_of_permute):
            save_name = save_path + image_name + "_%04d.png" % i
            print '\t test image {}'.format(save_name)
            if os.path.exists(save_name) and False:
                continue

            label_db_file = permute_path + image_name + "_%04d.png" % i
            pose_in_file = permute_path + image_name + "_%04d.txt" % i
            label_db = cv2.imread(label_db_file, cv2.IMREAD_UNCHANGED)
            label_db[label_db == 255] = 0
            pose_in = np.loadtxt(pose_in_file)

            inputs = OrderedDict([(data_names[0], image),
                      (data_names[1], label_db),
                      (data_names[2], pose_in)])

            res = infer(model, inputs, data_config, out_names)
            pose_out = res[out_names[0]]
            label_rect = data_iter.render_from_3d(pose_out, proj,
                    intr, params['color_map'],
                    is_color=params['is_color_render'],
                    label_map=params['id_2_trainid'])[0]

            # pose_gt = np.loadtxt(params['pose_path'] + image_name + '.txt')
            # pose_eval_metric.update([mx.nd.array(pose_gt[None, :])],
            #                         [mx.nd.array(pose_out)])

            # logging.info('Eval pose {}'.format(pose_eval_metric.get()))

            np.savetxt(save_name[:-4] + '.txt', res[out_names[0]])
            is_saved = cv2.imwrite(save_name, label_rect)
            if not is_saved:
                raise ValueError('can not write to %s' % save_name)

            # pose_gt = np.loadtxt(gt_path + image_name + '.txt')
            # print res
            # print previous_pose
            # print pose_gt

            # bkg_color = [255, 255, 255]
            # label_db_c = uts.label2color(label_db, params['color_map_list'],
            #         bkg_color)
            # label_rect_c = uts.label2color(label_rect, params['color_map_list'],
            #         bkg_color)
            # label_rect_pc = uts.label2color(label_rect, params['color_map_list'],
            #         bkg_color)
            # diff = label_rect - previous_res
            # pdb.set_trace()
            # uts.plot_images({'image':image,
            #     'label_db': label_db_c,
            #     'label_rect': label_rect_c,
            #     'label_rect_p':label_rect_pc,
            #     'diff_mask': diff}, layout=[2, 3])


def test_rnn():
    # set some parameters read image & pre render image
    ctx = mx.gpu(int(args.gpu_ids))
    np.set_printoptions(precision=3, suppress=True)
    camera_source = 'Camera_1'
    # res_source = 'posenet-proj-v2-0074'
    res_source = 'posenet-sem-wei-proj-0010'
    # unit_args = {'is_highorder': True, 'layer_num': 1}
    unit_args = {'is_highorder': True, 'layer_num': 2}
    is_render=True

    params = zpark.set_params()
    method = args.test_model.split('/')[-1]

    res_path = params['output_path'] + res_source + '/'
    save_path = params['output_path'] + res_source + '/' + method + '/'
    uts.mkdir_if_need(save_path)

    data_config, label_config = data_iter.get_pose_setting(
         with_points=False, with_pose_in=True)
    height, width = data_config['image']['size']
    all_images = [line.strip() for line in open(params['test_set']) \
         if camera_source in line]

    image_num = len(all_images)
    data_names = ['pose_in_%03d'%i for i in range(image_num)]
    data_shapes = [tuple([1, 6]) for i in range(image_num)]

    # define a sequence
    in_model = {'model': args.test_model}
    net, out_names = def_model(data_names, params, version='rnn',
            args=unit_args)
    model = nuts.load_model(in_model, data_names=data_names,
            data_shapes=data_shapes, net=net, ctx=ctx)

    # read onescene pose into a sequence
    pose_in = []
    for count, image_name in enumerate(all_images):
        trans_file = res_path + image_name + '_rect.txt'
        cur_pose = np.loadtxt(trans_file)
        cur_pose = data_iter.pose_transform(cur_pose)
        pose_in.append(mx.nd.array(cur_pose))

    res = iter_infer(model, Batch(pose_in), out_names)
    pose_gt_path = params['output_path'] + 'Loc/'
    pose_eval_metric = metric.PoseMetric(is_euler=True)
    seg_eval_metric = metric.SegMetric(ignore_label=0)

    if is_render:
        proj, intrinsic = data_iter.get_projector(params, height, width)

    for (key, pose_out), image_name in zip(res.items(), all_images):
        motion_gt = np.loadtxt(pose_gt_path + image_name + '.txt')
        pose_eval_metric.update([mx.nd.array(motion_gt[None, :])],
                [mx.nd.array(pose_out)])
        logging.info('Eval pose {}'.format(pose_eval_metric.get()))

        if is_render:
            ext_pred = data_iter.angle_to_proj_mat(
                    pose_out[0, :3], pose_out[0, 3:], is_quater=False)
            label_proj_c = proj.pyRenderToMat(intrinsic, ext_pred)
            label_proj = data_iter.color_to_id(
                    label_proj_c, height, width, params['color_map'])

            label_gt = cv2.imread(params['label_bkg_path'] + image_name + '.png',
                    cv2.IMREAD_UNCHANGED)
            label_gt = mx.nd.array(data_iter.label_transform(label_gt))

            # pose metric & segment metric
            seg_output = mx.nd.array(label_proj[None, :, :])
            seg_output = mx.ndarray.one_hot(seg_output, params['class_num'])
            seg_output = mx.ndarray.transpose(seg_output, axes=(0, 3, 1, 2))
            seg_eval_metric.update(label_gt, [seg_output])
            logging.info('Eval seg {}'.format(seg_eval_metric.get()))

        if args.is_save:
            test_scene = '/'.join(image_name.split('/')[:-1])
            uts.mkdir_if_need(save_path + test_scene)
            if is_render:
                cv2.imwrite(save_path+ image_name + '.png', label_proj)
            np.savetxt(save_path + image_name + '.txt', pose_out[0, :])

    logging.info('Final pose eval {}'.format(str(pose_eval_metric.get())))
    if is_render:
        logging.info('Final seg eval {}'.format(str(seg_eval_metric.get())))


class NaiveKalmanFilter(object):
    """We calculate speed and angle from testing data directly
    """
    def __init__(self, frames=10, sigma=1e-1, threshold=3):
        self._frames=frames
        self._sigma=sigma
        self._threshold=threshold

    def smooth_first(self, seq, reg='l2'):
        n, k = seq.shape
        x_t = seq[0, :]
        var_x = 1

        # only smooth straight line
        smoothed_seq = np.zeros((n, k))
        smoothed_seq[0, :] = seq[0, :]

        for i in range(1, n):
            # check whether need to memorize
            x_last = x_t.copy()
            if np.linalg.norm(seq[i, :] - x_t) > self._threshold:
                x_t = seq[i, :]
                var_x = 1
            else:
                x_t = (x_t * var_x + seq[i, :])/(var_x + 1)
                if reg == 'l1':
                    if np.linalg.norm(x_t - x_last) < self._sigma:
                        x_t = x_last
                var_x = (var_x + 1)

            smoothed_seq[i, :] = x_t

        return smoothed_seq


    def smooth_second(self, seq, reg='l2'):
        """We use the smooth over the second order,
           joint using camera and seq for smoothing
        Input:
            seq: [n x k] a seqence with n time steps
        """
        n, k = seq.shape
        x_t = seq[0, :]
        var_x = 0

        v_t = (seq[1, :] - seq[0, :])
        var_v = 0

        # only smooth straight line
        smoothed_seq = np.zeros((n, k))
        smoothed_seq[0, :] = seq[0, :]

        for i in range(1, n):
            # check whether need to memorize
            v_last = v_t.copy()
            v_cur = (seq[i, :] - seq[i - 1, :])

            if np.linalg.norm(v_cur - v_last) > self._threshold:
                v_t = v_cur
                var_v = 1
                var_x = 1
            else:
                # do prediction
                v_t = (v_t * var_v + v_cur) / (var_v + 1)
                var_v = (var_v + 1)

            # check whether need to do reduction
            if reg == 'l2':
                x_t = ((x_t + v_t) * var_x + seq[i, :])/(var_x + 1)
            else:
                if np.linalg.norm(v_t - v_last) < self._sigma:
                    x_t = (x_t + v_t)
                else:
                    x_t = ((x_t + v_t) * var_x + seq[i, :])/(var_x + 1)

            var_x = (var_x + 1)
            smoothed_seq[i, :] = x_t

        return smoothed_seq



def eval_pose_results(dataset, res_source=None, save_file=None, ):
    # eval the pre-rendered results
    logging.info(res_source)
    params = data_libs[dataset].set_params()

    if save_file is None:
        if res_source is None:
            save_file = params['sim_path']
        else:
            save_file = params['output_path'] + res_source + '/sim_test/'

    all_images = [line.strip() for line in open(params['test_set'])]
    sim_num = 1

    res_pose = []
    res_seg = []
    ext_path = 'LocPermute/' if res_source is None \
                else 'LocRect/%s/'%res_source
    label_db_path = params['output_path'] + ext_path

    # multi-time simulation test
    for i in range(sim_num):
        logging.info('%d/%d'%(i, sim_num))
        res_path = params['sim_path'] + '%02d/' % i
        save_file = save_file + '%02d/' % i
        rand_id = np.loadtxt(res_path + 'rand_id.txt')

        pose_eval_metric = metric.PoseMetric(is_euler=True)
        seg_eval_metric = metric.SegMetric(ignore_label=0)
        assert len(rand_id.tolist()) == len(all_images)

        for j, image_name in enumerate(all_images):
            if j % 10 == 0:
                logging.info('\t%d/%d'%(j, len(all_images)))

            idx = '_%04d' % int(rand_id[j])
            if res_source is None:
                res_file = '%s/%s.txt' % (res_path, image_name)
                label_db_file = '%s/%s%s.png' % (label_db_path,
                                 image_name, idx)
            else:
                res_file = '%s/%s%s.txt' % (label_db_path, image_name,
                        idx)
                label_db_file = '%s/%s%s.png' % (label_db_path,
                        image_name, idx)

            pred = np.loadtxt(res_file)
            label_db = cv2.imread(label_db_file, cv2.IMREAD_UNCHANGED)
            sz = label_db.shape

            pose_gt = np.loadtxt(params['pose_path'] + image_name + '.txt')
            label_gt = cv2.imread(params['label_bkgfull_path'] + image_name + '.png',
                    cv2.IMREAD_UNCHANGED)

            label_gt = cv2.resize(label_gt, (sz[1], sz[0]),
                                  interpolation=cv2.INTER_NEAREST)
            label_gt = mx.nd.array(data_iter.label_transform(label_gt))

            # pose metric & segment metric
            seg_output = mx.nd.array(label_db[None, :, :])
            seg_output = mx.ndarray.one_hot(seg_output, params['class_num'])
            seg_output = mx.ndarray.transpose(seg_output, axes=(0, 3, 1, 2))
            pose_eval_metric.update([mx.nd.array(pose_gt[None, :])],
                    [mx.nd.array(pred[None, :])])
            seg_eval_metric.update(label_gt, [seg_output])

        pose_name, pose_value = pose_eval_metric.get()
        seg_name, seg_value = seg_eval_metric.get()
        res_pose.append(pose_value)
        res_seg.append(seg_value)

    res_pose = np.array(res_pose)
    res_seg = np.array(res_seg)
    print pose_name, np.mean(res_pose, axis=0), np.std(res_pose, axis=0)
    print seg_name, np.mean(res_seg, axis=0), np.std(res_seg, axis=0)

    np.savetxt(save_file + '_pose.txt', res_pose)
    np.savetxt(save_file + '_seg.txt', res_seg)


def eval_pose_rnn_results(method=None,
                          save_file=None,
                          posecnn='',
                          dataname='dlake'):

    # eval the pre-rendered results
    logging.info('%s with %s ' % (method, posecnn))
    params = eval('data.' + dataname + '.set_params()')

    if 'noise' in method:
        if dataname == 'zpark':
            ext_path = 'LocPermute/'
            out_num = 7
            is_quater = True
        elif dataname == 'dlake':
            ext_path = params['pose_permute']
            out_num = 6
            is_quater = False
    else:
        ext_path = params['pose_rect'] + posecnn + '/'
        out_num = 6
        is_quater = False

    if save_file is None:
        save_file_root = params['output_path'] + method + '/sim_test/'

    height, width = params['in_size']
    proj = data_iter.get_projector(params, height, width)

    pose_model = {'model':'./Output/' + method}
    _, arg_params, aux_params = nuts.args_from_models(pose_model, False)

    # load seqence
    image_list_all = [line.strip() for line in open(params['test_set'])]
    res_pose = []
    res_seg = []
    sim_num = 10

    for i in range(sim_num):
        logging.info('testing %d/%d'%(i, sim_num))
        res_path = params['sim_path'] + '%02d/' % i
        save_file = save_file_root + '%02d/' % i
        rand_id = np.loadtxt(res_path + 'rand_id.txt')

        assert len(rand_id) == len(image_list_all)
        image_seq = data_iter.image_set_2_seqs(image_list_all, params['cam_names'],
                rand_idx=rand_id)
        logging.info('test seqs len %d' % len(image_seq))
        pdb.set_trace()

        pose_eval_metric = metric.PoseMetric(is_euler=(not is_quater))
        seg_eval_metric = metric.SegMetric(ignore_label=0)

        # directly apply rnn here
        for count, image_list in enumerate(image_seq):
            logging.info('predict sequence %d/%d, len %d' % (
                count, len(image_seq), len(image_list)))

            # predict
            image_num = len(image_list)
            data_names = ['pose_in_%03d' % i for i in range(image_num)]
            data_shapes = [tuple([1, out_num]) for i in range(image_num)]
            data_shapes = [(name, shape) for name, shape \
                    in zip(data_names, data_shapes)]

            inputs = nuts.get_mx_var_by_name(data_names)
            net = pose_nn.recurrent_pose(inputs, name='pose',
                    out_num=out_num, is_train=False,
                    is_highorder=True, layer_num=2)

            net = mx.sym.Group(net.values())
            mod = mx.mod.Module(net,
                                data_names=data_names,
                                label_names=None,
                                context=mx.gpu(int(args.gpu_ids)))

            mod.bind(for_training=False, data_shapes=data_shapes)
            mod.set_params(arg_params,
                           aux_params,
                           allow_missing=False)
            # read
            pose_in = []
            for image_name, idx in image_list:
                trans_file = '%s/%s_%04d.txt' % (ext_path, image_name, idx)
                cur_pose = np.loadtxt(trans_file)

                if is_quater:
                    cur_pose = np.concatenate([cur_pose[:3],
                            uts_3d.euler_angles_to_quaternions(
                            cur_pose[3:6])])
                cur_pose = data_iter.pose_transform(cur_pose)
                pose_in.append(mx.nd.array(cur_pose))

            mod.forward(Batch(pose_in))
            output_nd = mod.get_outputs()
            for res, (image_name, idx) in zip(output_nd, image_list):
                test_scene = '/'.join(image_name.split('/')[:-1])
                save_path = save_file + test_scene
                uts.mkdir_if_need(save_path)
                pose_out = res.asnumpy()[0]

                if is_quater:
                    pose_out[3:] = pose_out[3:]/np.linalg.norm(
                            pose_out[3:])

                # save results
                np.savetxt(save_file + image_name + '.txt', pose_out)
                ext = data_iter.angle_to_proj_mat(
                        pose_out[:3], pose_out[3:], is_quater=is_quater)
                intr_key = data_iter.get_intr_key(params['cam_names'], image_name)
                intr = data_iter.get_proj_intr(params['intrinsic'][intr_key],
                        height, width)
                label_db = data_iter.render_from_3d(pose_out[None, :], proj,
                        intr, params['color_map'],
                        is_color=params['is_color_render'],
                        label_map=params['id_2_trainid'])[0]

                cv2.imwrite(save_file + image_name + '.png', label_db)
                pose_gt = np.loadtxt(params['pose_path'] + image_name + '.txt')
                if is_quater:
                    pose_gt = np.concatenate([pose_gt[:3],
                            uts_3d.euler_angles_to_quaternions(
                            pose_gt[3:6])])
                label_gt = cv2.imread(params['label_bkgfull_path'] + \
                               image_name + '.png', cv2.IMREAD_UNCHANGED)
                label_gt = cv2.resize(label_gt, (width, height),
                               interpolation=cv2.INTER_NEAREST)
                label_gt = mx.nd.array(data_iter.label_transform(label_gt))

                # pose metric & segment metric
                seg_output = mx.nd.array(label_db[None, :, :])
                seg_output = mx.ndarray.one_hot(
                        seg_output, params['class_num'])
                seg_output = mx.ndarray.transpose(
                        seg_output, axes=(0, 3, 1, 2))
                pose_eval_metric.update([mx.nd.array(pose_gt[None, :])],
                        [mx.nd.array(pose_out[None, :])])
                seg_eval_metric.update(label_gt, [seg_output])

                logging.info('Eval seg {}'.format(seg_eval_metric.get()))
                logging.info('Eval pose {}'.format(pose_eval_metric.get()))

        pose_name, pose_value = pose_eval_metric.get()
        seg_name, seg_value = seg_eval_metric.get()
        res_pose.append(pose_value)
        res_seg.append(seg_value)

    res_pose = np.array(res_pose)
    res_seg = np.array(res_seg)
    print pose_name, np.mean(res_pose, axis=0), np.std(res_pose, axis=0)
    print seg_name, np.mean(res_seg, axis=0), np.std(res_seg, axis=0)

    np.savetxt(save_file + 'sim_test_pose.txt', res_pose)
    np.savetxt(save_file + 'sim_test_seg.txt', res_seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the trained model for acc.')
    parser.add_argument('--model_path', default='./Output/', help='The model name with mxnet format.')
    parser.add_argument('--gpu_ids', default="1",
        help='the gpu ids use for training')
    parser.add_argument('--is_save', action="store_true", default=False,
        help='whether to save the results')

    args = parser.parse_args()
    logging.info(args)
    # test_posenet()
    # test_posenet_v1('posecnn-proj-0094', 'dlake')
    # test_posenet_v1_demo('posecnn-test-0030')

    eval_pose_results('posenet-sem-wei-proj-0010')
    eval_pose_rnn_results('posenet_rnn_hier_euler-0061')



