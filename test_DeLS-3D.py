"""
Test demo for pretrained models.
Author: 'peng wang'

"""

import argparse
import cv2
import logging
import mxnet as mx
import numpy as np
import vis_metrics as metric
import time
import vis_utils as uts
import data_transform as ts
import utils_3d as uts_3d
import utils.dels_utils as de_uts

import dataset.data_iters as data_iter
import networks.pose_nn as pose_nn
import networks.net_util as nuts
from config import config
from collections import namedtuple, OrderedDict
import pdb
debug = 1

np.set_printoptions(precision=4, suppress=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
Batch = namedtuple('Batch', ['data'])


def infer(mod,
          inputs,
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


def test_pose_cnn_demo(method, dataname):
    logging.info(method)

    ctx = mx.gpu(int(args.gpu_ids))
    params = config.dataset[dataname].set_params(
                        config.path.data_root,
                        config.path.model_root)

    # define saving place
    data_setting, label_setting = config.pose_nn_setting(
                        with_pose_in=True)
    height, width = data_setting['image']['size']
    proj = data_iter.get_projector(params, height, width)

    # init model
    data_names = ['image', 'label_db', 'pose_in']
    data_shapes = [tuple([1, 3] + data_setting['image']['size']),
                   tuple([1, 1] + data_setting['label_db']['size']),
                   tuple([1, 6])]

    net, out_names = def_model(data_names, params, version='cnn')
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
        label_gt = cv2.imread(params['label_bkg_path'] + \
                image_name + '.png',
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


class DeLS3D(object):
    def __init__(self, dataname,
                 pose_cnn,
                 pose_rnn,
                 seg_cnn,
                 is_save_inter=False):
        """
        Performing deep localization and segmentation on validation set
        of corresponding data
        """
        # targeting at specific scenes with 3D point cloud
        assert dataname in ['zpark', 'dlake']
        from dataset import data_lib
        self.params = data_lib[dataname].set_params()
        print(self.params['size'])
        self.renderer = de_uts.Renderer(self.params,
                                        self.params['cloud'],
                                        self.params['size'])
        self.models = {'pose_cnn': pose_cnn,
                       'pose_rnn': pose_rnn,
                       'seg_cnn': seg_cnn}

        self.noisy_path = None
        self.sim_noisy_params = {}
        self.ctx = mx.gpu(int(args.gpu_ids))
        self.test_image_list = [line.strip() for line in \
                open(self.params['test_set'])]


    def def_model(self, in_names, version='cnn', args=None):
        """ set up the symbolic for data
        """

        inputs = nuts.get_mx_var_by_name(in_names)
        if version == 'cnn':
            net = pose_nn.pose_block(inputs,
                        self.params,
                        in_names=in_names,
                        name='pose',
                        is_train=False)

        elif version == 'rnn':
            net = pose_nn.recurrent_pose(inputs, name='pose',
                    is_train=False, **args)

        else:
            raise ValueError('no such network')

        return mx.symbol.Group(net.values()), net.keys()

    def get_sim_seq(self, re_generate=True):
        """ Use a sample sequence pregenerated
        """
        self.noisy_pose_path = self.params['noisy_pose_path']
        if re_generate:
            for image_name in self.test_image_list:
                pose_gt_file = '%s/%s.txt' % (self.params['pose_path'],
                        image_name)
                pose_gt = de_uts.loadtxt(pose_gt_file)
                trans_i, rot_i = uts_3d.random_perturb(pose_gt[:3],
                        pose_gt[3:], 5.0)
                noisy_pose = np.concatenate([trans_i, rot_i])
                noisy_pose_file = '%s/%s.txt' % (self.noisy_pose_path,
                        image_name)
                np.savetxt(noisy_pose_file, noisy_pose)
        self.render_segments(self.noisy_pose_path)

        return self.params['noisy_pose_path']


    def localize_and_segment(self):
        """
           self.pose_seq: sequence of noisy GPS/IMU
           one may also adjust the
        """
        self.pose = self.refine_with_pose_cnn()
        self.pose = self.refine_with_pose_rnn()
        self.get_segment(self.pose_rnn_output_path)


    def refine_with_pose_cnn(self, render_segment=False):
        # first stage do perimage refining
        self.pose_cnn_output_path = '%s/pose_cnn/' % self.params['output_path']

        # define saving place
        data_setting, label_setting = config.network.pose_cnn_setting(
                                        with_pose_in=True)
        height, width = data_setting['image']['size']

        # init model
        data_names = ['image', 'label_db', 'pose_in']
        data_shapes = [tuple([1, 3] + data_setting['image']['size']),
                       tuple([2, 1] + data_setting['label_db']['size']),
                       tuple([1, 6])]

        net, out_names = self.def_model(data_names, version='cnn')
        model = nuts.load_model(self.models['pose_cnn'], data_names=data_names,
                                data_shapes=data_shapes, net=net, ctx=self.ctx)

        for i, image_name in enumerate(self.test_image_list):
            logging.info('%d/%d, %s'%(i, len(self.test_image_list), image_name))

            # get input
            image_file = '%s/%s.jpg' % (self.params['image_path'], image_name)
            image = de_uts.imread(image_file)
            pose_file = '%s/%s.txt' % (self.noisy_pose_path, image_name)
            pose_in = de_uts.loadtxt(pose_file)
            proj_label_file = '%s/%s.png' % (self.noisy_pose_path, image_name)
            label_db = de_uts.imread(proj_label_file)

            inputs = OrderedDict([('image', image),
                                  ('label_db', label_db),
                                  ('pose_in', pose_in)])
            # get pose and corresponding projected map
            res = infer(model, inputs, data_setting, out_names)
            pose_out = res[out_names[0]]

            self._get_save_path(self.pose_cnn_output_path, image_name)
            np.savetxt('%s/%s.txt' % (self.pose_cnn_output_path, image_name),
                       pose_out)

        if render_segment:
            self.render_segments(self.pose_cnn_output_path)


    def refine_with_pose_rnn(self, render_segment=False):
        """refine the output from pose_cnn using pose_rnn
        """
        pass
        # self.pose_cnn_output_path = '%s/pose_rnn/' % params['output_path']
        # res_path = params['sim_path'] + '%02d/' % i
        # save_file = save_file_root + '%02d/' % i
        # rand_id = np.loadtxt(res_path + 'rand_id.txt')

        # assert len(rand_id) == len(image_list_all)
        # image_seq = data_iter.image_set_2_seqs(image_list_all, params['cam_names'],
        #                                        rand_idx=rand_id)
        # logging.info('test seqs len %d' % len(image_seq))

        # _, arg_params, aux_params = nuts.args_from_models(pose_model, False)
        # for count, image_list in enumerate(image_seq):
        #     logging.info('predict sequence %d/%d, len %d' % (
        #         count, len(image_seq), len(image_list)))

        #     pose_in = []
        #     for image_name, idx in image_list:
        #         trans_file = '%s/%s_%04d.txt' % (ext_path, image_name, idx)
        #         cur_pose = np.loadtxt(trans_file)

        #         if is_quater:
        #             cur_pose = np.concatenate([cur_pose[:3],
        #                     uts_3d.euler_angles_to_quaternions(
        #                     cur_pose[3:6])])
        #         cur_pose = data_iter.pose_transform(cur_pose)
        #         pose_in.append(mx.nd.array(cur_pose))

        #     # predict
        #     image_num = len(image_list)
        #     data_names = ['pose_in_%03d' % i for i in range(image_num)]
        #     data_shapes = [tuple([1, out_num]) for i in range(image_num)]
        #     data_shapes = [(name, shape) for name, shape \
        #             in zip(data_names, data_shapes)]

        #     inputs = nuts.get_mx_var_by_name(data_names)
        #     net = pose_nn.recurrent_pose(inputs, name='pose',
        #             out_num=out_num, is_train=False,
        #             is_highorder=True, layer_num=2)

        #     net = mx.sym.Group(net.values())
        #     mod = mx.mod.Module(net,
        #                         data_names=data_names,
        #                         label_names=None,
        #                         context=mx.gpu(int(args.gpu_ids)))

        #     mod.bind(for_training=False, data_shapes=data_shapes)
        #     mod.set_params(arg_params,
        #                    aux_params,
        #                    allow_missing=False)
        #     # read

        #     mod.forward(Batch(pose_in))
        #     output_nd = mod.get_outputs()
        #     for res, (image_name, idx) in zip(output_nd, image_list):
        #         test_scene = '/'.join(image_name.split('/')[:-1])
        #         save_path = save_file + test_scene
        #         uts.mkdir_if_need(save_path)
        #         pose_out = res.asnumpy()[0]

        #         if is_quater:
        #             pose_out[3:] = pose_out[3:]/np.linalg.norm(
        #                     pose_out[3:])

        #         # save results
        #         np.savetxt(save_file + image_name + '.txt', pose_out)
        #         ext = data_iter.angle_to_proj_mat(
        #                 pose_out[:3], pose_out[3:], is_quater=is_quater)
        #         intr_key = data_iter.get_intr_key(params['cam_names'], image_name)
        #         intr = data_iter.get_proj_intr(params['intrinsic'][intr_key],
        #                 height, width)
        #         label_db = data_iter.render_from_3d(pose_out[None, :], proj,
        #                 intr, params['color_map'],
        #                 is_color=params['is_color_render'],
        #                 label_map=params['id_2_trainid'])[0]

        #         cv2.imwrite(save_file + image_name + '.png', label_db)
        #         pose_gt = np.loadtxt(params['pose_path'] + image_name + '.txt')
        #         if is_quater:
        #             pose_gt = np.concatenate([pose_gt[:3],
        #                     uts_3d.euler_angles_to_quaternions(
        #                     pose_gt[3:6])])
        #         label_gt = cv2.imread(params['label_bkgfull_path'] + \
        #                        image_name + '.png', cv2.IMREAD_UNCHANGED)
        #         label_gt = cv2.resize(label_gt, (width, height),
        #                        interpolation=cv2.INTER_NEAREST)
        #         label_gt = mx.nd.array(data_iter.label_transform(label_gt))

        #         # pose metric & segment metric
        #         seg_output = mx.nd.array(label_db[None, :, :])
        #         seg_output = mx.ndarray.one_hot(
        #                 seg_output, params['class_num'])
        #         seg_output = mx.ndarray.transpose(
        #                 seg_output, axes=(0, 3, 1, 2))
        #         pose_eval_metric.update([mx.nd.array(pose_gt[None, :])],
        #                 [mx.nd.array(pose_out[None, :])])
        #         seg_eval_metric.update(label_gt, [seg_output])

        #         logging.info('Eval seg {}'.format(seg_eval_metric.get()))
        #         logging.info('Eval pose {}'.format(pose_eval_metric.get()))

    def get_segment(self, pose_path):
        pass



    def _get_save_path(self, output_path, image_name):
        """ get the corresponding saving path from image name in apolloscape
            dataset
        """

        test_scene = '/'.join(image_name.split('/')[:-1])
        save_path = '%s/%s/' % (output_path, test_scene)
        uts.mkdir_if_need(save_path)

        return save_path


    def render_segments(self, pose_path):
        """Render labelled image out from generated poses
        """
        for i, image_name in enumerate(self.test_image_list):
            pose_file = '%s/%s.txt' % (self.noisy_pose_path, image_name)
            pose_in = de_uts.loadtxt(pose_file)
            intr_key = data_iter.get_intr_key(self.params['intrinsic'].keys(),
                                              image_name)
            label_db = self.renderer.render_from_3d(pose_in[None, :],
                                         self.params['intrinsic'][intr_key],
                                         self.params['color_map'],
                                         is_color=self.params['is_color_render'],
                                         label_map=self.params['id_2_trainid'])

            save_path = self._get_save_path(pose_path, image_name)
            cv2.imwrite('%s/%s.png' % (save_path, image_name), label_db)
            if debug:
                uts.plot_images({'label_db': label_db})


    def get_refined_pose_path(self):
        return self.pose_cnn_output_path

    def get_refined_segment_path(self):
        return self.seg_cnn_output_path


    def eval_pose(self, pose_res_path, eval_render_segment=False):
        """
        with_segment: if also evaluate projected segments
        """
        # start eval
        pose_eval_metric = metric.PoseMetric(is_euler=True)

        for image_name in self.test_image_list:
            gt_file = self.params['pose_path'] + image_name + '.txt'
            pose_gt = de_uts.loadtxt(gt_file)
            res_file = '%s/%s.txt' % (pose_res_path + image_name)
            pose_out = de_uts.loadtxt(res_file)
            pose_eval_metric.update([mx.nd.array(pose_gt[None, :])],
                                    [mx.nd.array(pose_out)])

            logging.info('Eval pose {}'.format(pose_eval_metric.get()))

        logging.info('Final pose eval {}'.format(str(pose_eval_metric.get())))

        if eval_render_segment:
            self.eval_segments(seg_res_path)


    def eval_segments(self, seg_res_path):
        seg_eval_metric = metric.SegMetric(ignore_label=255)
        for image_name in self.test_image_list:
            # evaluation pose metric & segment metric
            gt_file = '%s/%s.png' % (self.params['label_path'], image_name)
            label_gt = de_uts.imread(gt_file)
            res_file = '%s/%s.png' % (seg_res_path, image_name)
            label_res = de_uts.imread(res_file)
            label_seg = mx.nd.array(ts.label_transform(label_gt))
            seg_output = mx.nd.array(label_res[None, :, :])
            seg_output = mx.ndarray.one_hot(seg_output, params['class_num'])
            seg_output = mx.ndarray.transpose(seg_output, axes=(0, 3, 1, 2))
            seg_eval_metric.update(label_seg, [seg_output])


    def save_frame_to_video(self, seg_res_path):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the trained model for acc.')
    parser.add_argument('--data', default='dlake',
            help='The dataset name with mxnet.')
    parser.add_argument('--gpu_ids', default="0",
            help='the gpu ids use for training')
    parser.add_argument('--pose_cnn', default='',
            help='The model name with mxnet format.')
    parser.add_argument('--pose_rnn', default='',
            help='The model name with mxnet format.')
    parser.add_argument('--seg_cnn', default='',
            help='The model name with mxnet format.')

    args = parser.parse_args()
    logging.info(args)

    # test_pose_cnn_demo('posecnn-test-0030', 'dlake')
    # test_posenet_v1('posecnn-proj-0094', 'dlake')

    # then load the trained networks cnn rnn seg_cnn,
    DeLS = DeLS3D(args.data,
                  args.pose_cnn,
                  args.pose_rnn,
                  args.seg_cnn)

    # first simulate a perturbation sequence using args.dataset
    # pose_seq = uts.simulate_permuted_sequence(
    #                              args.dataset, split='val')

    # get the sequence pre-simulated
    save_path = DeLS.get_sim_seq()
    DeLS.localize_and_segment(sim_pose_path=save_path)

    # if we have ground truth for evaluation.
    DeLS.eval_pose(DeLS.get_refined_pose_path(),
              DeLS.get_gt_pose_path())
    DeLS.eval_segments(DeLS.get_segment_path(),
              DeLS.get_gt_segment_path())

    # save the images to video for visualization




