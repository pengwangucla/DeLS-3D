"""Test demo for trained model
"""
import os
import argparse
import logging

import cv2
import mxnet as mx
import numpy as np
import time
import utils.utils as uts
import utils.metric as metric

import Data.dataset as data
import Data.data_iters as data_iter
import Networks.mxvideo_segnet as segnet
import Networks.net_util as nuts
import Networks.augmentor as aug
from collections import namedtuple, OrderedDict

np.set_printoptions(precision=5, suppress=True)
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
          out_names,
          augmentor=None,
          eval_metric=None,
          label=None,
          label_setting=None):

    data_dict = OrderedDict([])
    s = time.time()
    for data_name in inputs:
        data = inputs[data_name]
        data = np.array(data, dtype=np.float32)  # (h, w, c)

        data_aug = np.expand_dims(data, axis=0)
        if augmentor is not None:
            data_aug = augmentor.augment(data)
        is_img = 'size' in data_setting[data_name]
        if is_img:
            height, width = data_setting[data_name]['size']
            interpolation = data_setting[data_name]['resize_method']

        transform = data_setting[data_name]['transform']

        data_this = []
        for data in data_aug:
            if is_img:
                data = cv2.resize(data, (width, height),
                                  interpolation=interpolation)
            data_this.append(transform(data))
        data_dict[data_name] = np.concatenate(data_this, axis=0)

    data_list = [mx.nd.array(data_dict[data_name]) \
                 for data_name in data_dict]
    prep_time = time.time() - s

    mod.forward(Batch(data_list))
    output_nd = mod.get_outputs()
    forw_time = time.time() - prep_time - s

    output = OrderedDict([])
    for i, name in enumerate(out_names):
        output_nd[i].wait_to_read()
        asnp_time = time.time() - prep_time - s - forw_time
        np_array = output_nd[i].asnumpy()
        cur_out = np.transpose(np_array, [0, 2, 3, 1])

        # pdb.set_trace()
        if augmentor is not None:
            merge_out = augmentor.merge(cur_out)
            output[name] = to_prob(merge_out, axis=2)
        else:
            output[name] = np.squeeze(cur_out)

    # post_time = time.time() - forw_time - prep_time - s- asnp_time
    # logging.info('forward time pre {}, forw{}, asnp{}, post {}'.format(
    #     prep_time, forw_time, asnp_time, post_time))

    if eval_metric:
        label_name = label_setting.keys()[0]
        height, width = label_setting[label_name]['size']
        interpolation = label_setting[label_name]['resize_method']
        transform = label_setting[label_name]['transform']
        label = cv2.resize(label, (width, height),
                           interpolation=interpolation)
        label = mx.nd.array(transform(label))
        assert len(output_nd) == 1
        eval_output = mx.nd.array(np.expand_dims(
            np.transpose(output[name], [2, 0, 1]), axis=0))
        eval_metric.update([label], [eval_output])
        logging.info('Eval {}'.format(eval_metric.get()))

    return output


def def_model_segnet(params, net_name='segment', has_3d=False,
        is_refine=False, spatial_step=1, suppress=1):
    input = dict([('data', mx.symbol.Variable(name="data"))])
    ext_inputs = dict([('label_db', mx.symbol.Variable(name="label_db"))])

    block = segnet.segment_block(input, params, name=net_name,
            ext_inputs=ext_inputs, is_train=False)
    arg_params = nuts.def_arguments(block[net_name + '_score'].list_arguments())

    in_name = ['data', 'label_db'] if has_3d else ['data']
    net = segnet.recurrent_seg_block(input,
                                     params,
                                     name=net_name,
                                     in_names=in_name,
                                     ext_inputs=ext_inputs,
                                     spatial_steps=spatial_step,
                                     arg_params=arg_params,
                                     is_refine=is_refine,
                                     is_train=False,
                                     suppress=suppress)
    return mx.symbol.Group(net.values()), net.keys()


def def_model_pose_segnet(data_names, params,
        net_name='segment', suppress=1):

    input = nuts.get_mx_var_by_name(data_names)
    net = segnet.pose_seg_net(input, params,
            name=net_name,
            is_train=False,
            suppress=suppress)
    return mx.symbol.Group(net.values()), net.keys()


def net_config(net_name):
    net_conf ={}
    net_conf['has_proj'] = False
    net_conf['has_posenet'] = False
    net_conf['suppress'] = False
    net_conf['pre_render'] = False
    return net_conf


def get_label_path(gt_type, params):
    if gt_type == 'bkgobj':
        return params['label_bkgobj_path']
    else:
        return params['label_path']



def test_seg_multi(is_refine, res_source, gt_type, is_rnn, dataname):
    """test a segment network with random pose as input
    """

    suppress=1
    rerun = True
    ctx = mx.gpu(int(args.gpu_ids))
    params = eval('data.' + dataname + '.set_params()')
    method = args.test_model.split('/')[-1]
    save_path_root = params['output_path'] + method + '/sim_test/'
    print res_source, gt_type, is_rnn

    # setting the experiment
    data_config, label_config = data_iter.get_seg_setting_v2(True)
    height, width = data_config['data']['size']
    data_names = ['data', 'label_db']
    data_shapes = [tuple([1, 3] + data_config[name]['size']) for name in ['data']]
    params['batch_size'] = 1
    data_shapes = data_shapes + [tuple([1] + data_config['label_db']['size'])]

    augmentor = None
    if args.is_augment:
        augmentor = aug.TestAugmentor(['flip']) # set default parameters

    net, out_names = def_model_segnet(params, args.net_name,  True,
                                      is_refine=is_refine,
                                      suppress=suppress)

    models = {'model': args.test_model}
    model = nuts.load_model(models, data_names, data_shapes, net, ctx)
    image_list = [line.strip() for line in open(params['test_set'])]

    ignore_labels = [0, 255]
    if gt_type == 'bkgobj':
        ignore_labels = [i for i in range(params['class_num']) if \
                    (not params['is_exist'][i]) \
                     or params['is_obj'][i] \
                     or params['is_rare'][i] \
                     or params['is_unknown'][i]]
        ignore_labels.remove(32)

    if is_rnn:
        proj = data_iter.get_projector(params, height, width)

    time_all = 0.
    sim_num = 1
    label_db_path = params['pose_permute'] if res_source is None \
                else  '%s/%s/' % (params['output_path'], res_source)

    res_seg = []
    res_ious = []

    for i in range(sim_num):
        logging.info('testing %d/%d'%(i, sim_num))
        res_path = params['sim_path'] + '%02d/' % i
        save_path = save_path_root + '%02d/' % i
        rand_id = np.loadtxt(res_path + 'rand_id.txt')
        assert len(rand_id) == len(image_list)

        eval_metric = metric.SegMetric(ignore_label=ignore_labels)
        if is_rnn:
            ext_path = '%s/sim_test/%02d/' % (res_source, i)
            label_db_path = params['output_path'] + ext_path

        for count, image_name in enumerate(image_list[:1892]):
            image_path = params['image_path'] + image_name + '.jpg'
            idx = '_%04d' % int(rand_id[count])
            label_db_file = '%s/%s%s.png' % (label_db_path, image_name, idx)

            if count % 10 == 0:
                logging.info('\t%d/%d'%(count, len(image_list)))

            # load the predicted pose
            scene = '/'.join(image_name.split('/')[:-1])
            uts.mkdir_if_need(save_path + scene)
            print '\t segmenting image {}\n'.format(image_name),

            if os.path.exists(save_path + image_name + '.png') and (not rerun):
                label_res = cv2.imread(save_path + image_name + '.png',
                        cv2.IMREAD_UNCHANGED)
                label_prob = uts.one_hot(label_res, params['class_num'])

            else:
                start_time = time.time()
                image = cv2.imread(image_path)
                inputs = {'data': image}
                if not is_rnn:
                    label_db = cv2.imread(label_db_file,
                                          cv2.IMREAD_UNCHANGED)

                if is_rnn:
                    label_db_file = '%s/%s.txt' % (label_db_path, image_name)
                    if not os.path.exists(label_db_file):
                        continue

                    pose_in = np.loadtxt(label_db_file)

                    intr_key = data_iter.get_intr_key(params['intrinsic'].keys(),
                            image_name)
                    intr = data_iter.get_proj_intr(params['intrinsic'][intr_key],
                            height, width)
                    label_db = data_iter.render_from_3d(pose_in[None, :],
                            proj, intr, params['color_map'],
                            is_color=params['is_color_render'],
                            label_map=params['id_2_trainid'])[0]

                    label_db_img = '%s/%s.png' % (label_db_path, image_name)
                    cv2.imwrite(label_db_img, label_db)

                inputs['label_db'] = label_db
                label_prob = infer(model, inputs,
                                   data_config,
                                   out_names,
                                   augmentor=augmentor)
                label_prob = label_prob[out_names[-1]]
                cost_time = time.time() - start_time
                time_all += cost_time

            # read gt to
            height, width = label_config['softmax_label']['size']
            label_path = get_label_path(gt_type, params)
            label_path = '%s%s.png' % (label_path, image_name)

            label_gt = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            label_gt = cv2.resize(label_gt, (width, height),
                                  interpolation=cv2.INTER_NEAREST)
            label_gt = data_iter.label_transform(label_gt,
                           gt_type=gt_type,
                           obj_ids=params['obj_ids'],
                           label_mapping=params['id_2_trainid'])
            label_gt_nd = mx.nd.array(label_gt)

            pred = mx.nd.array(np.expand_dims(
                      np.transpose(label_prob, [2, 0, 1]), axis=0))
            eval_metric.update([label_gt_nd], [pred])


            label = uts.prob2label(label_prob)

            # bkg_color = [255, 255, 255]
            # label_gt_c = uts.label2color(label_gt, params['color_map_list'],
            #         bkg_color)
            # label_c = uts.label2color(label, params['color_map_list'],
            #                           bkg_color)
            # label_db_c = uts.label2color(label_db, params['color_map_list'],
            #         bkg_color)
            # uts.plot_images({'image':image,
            #                  'label_db': label_db_c,
            #                  'label_pred': label_c,
            #                  'label_gt': label_gt_c})
            is_write = cv2.imwrite(save_path + image_name + '.png', label)
            if not is_write:
                raise ValueError('not able to write')

        name, res = eval_metric.get()
        logging.info('Final eval {} ave time {}'.format(str((name, res)), time_all / len(image_list)))
        res_seg.append(res)

        ious = eval_metric.get_ious()
        for i, class_name in enumerate(params['class_names']):
            print '%s: %05f \t' % (class_name, ious[i])

        res_ious.append(ious)

    res_seg = np.array(res_seg)
    res_ious = np.array(res_ious)

    print name, np.mean(res_seg, axis=0), np.std(res_seg, axis=0)
    np.savetxt(save_path_root + 'res_seg.txt', res_seg)
    np.savetxt(save_path_root + 'res_ious.txt', res_ious)


def test_seg(is_refine, res_source, gt_type, dataname='dlake'):
    """test a single network performance
    """
    suppress=1
    pre_render = True
    rerun = False

    ctx = mx.gpu(int(args.gpu_ids))
    params = eval('data.' + dataname + '.set_params()')
    data_config, label_config = data_iter.get_seg_setting_v2(args.has_3d)
    height, width = data_config['data']['size']

    data_names = ['data', 'label_db'] if args.has_3d else ['data']
    data_shapes = [tuple([1, 3] + data_config[name]['size']) for name in ['data']]
    params['batch_size'] = 1

    if args.has_3d:
        if not pre_render:
            proj, intr_for_render = data_iter.get_projector(
                    params, height, width)

        trans_ext = '%s/' % res_source
        data_shapes = data_shapes + [tuple([1] + data_config['label_db']['size'])]

    augmentor = None
    if args.is_augment:
        augmentor = aug.TestAugmentor(['flip']) # set default parameters

    net, out_names = def_model_segnet(params, args.net_name,
            args.has_3d, is_refine=is_refine,
            suppress=suppress)

    models = {'model': args.test_model}
    model = nuts.load_model(models, data_names, data_shapes, net, ctx)
    method = args.test_model.split('/')[-1]
    image_list = [line.strip() for line in open(params['test_set'])]

    ignore_labels = [0, 255]
    if gt_type == 'bkg':
        ignore_labels = [i for i in range(params['class_num']) if \
                    (not params['is_exist'][i]) \
                     or params['is_obj'][i] \
                     or params['is_rare'][i] \
                     or params['is_unknown'][i]]

    elif gt_type == 'bkgobj':
        ignore_labels = [i for i in range(params['class_num']) if \
                    (not params['is_exist'][i]) \
                     or params['is_rare'][i] \
                     or params['is_unknown'][i]]

    eval_metric = metric.SegMetric(ignore_label=ignore_labels)

    time_all = 0.
    for count, image_name in enumerate(image_list[:1892]):
        image_path = params['image_path'] + image_name + '.jpg'
        # load the predicted pose
        if args.has_3d:
            trans_path = params['output_path'] + trans_ext + image_name + '.txt'

        save_path = params['output_path'] + method + '/'
        scene = '/'.join(image_name.split('/')[:-1])
        uts.mkdir_if_need(save_path + scene)

        start_time = time.time()
        if os.path.exists(save_path + image_name + '.png') and rerun:
            label_res = cv2.imread(save_path + image_name + '.png',
                                   cv2.IMREAD_UNCHANGED)
            label_prob = uts.one_hot(label_res, params['class_num'])
        else:
            print '\t segmenting image {}'.format(image_name),
            image = cv2.imread(image_path)
            inputs = {'data' : image}
            if args.has_3d:
                if not pre_render:
                    trans = data_iter.trans_reader(
                            trans_path, proj_mat=True)
                    label_db = proj.pyRenderToMat(intr_for_render, trans)
                    label_db = data_iter.color_to_id(
                            label_db, height, width, params['color_map'])
                else:
                    if method is None:
                        label_db = data_iter.trans_reader_pre(trans_path)
                    else:
                        if res_source == 'gt':
                            gt_path = params['label_bkg_path'] + image_name + '.png'
                            # label_db is matched to res labels
                            label_db = cv2.imread(gt_path,
                                                  cv2.IMREAD_UNCHANGED)
                        else:
                            trans_path = trans_path[:-4] + '_label_rect.png'
                            label_db = cv2.imread(trans_path,
                                                  cv2.IMREAD_UNCHANGED)
                inputs['label_db'] = label_db

            label_prob = infer(model, inputs,
                               data_config,
                               out_names,
                               augmentor=augmentor)
            label_prob = label_prob[out_names[-1]]

        height, width = label_config['softmax_label']['size']
        label_path = params['label_path'] + image_name + '.png'

        if gt_type == 'bkgobj' or gt_type == 'binary':
            label_path = params['label_bkgobj_path'] + image_name + '.png'

        label_gt = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label_gt = cv2.resize(label_gt, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        label_gt = data_iter.label_transform(label_gt,
                       gt_type=gt_type,
                       obj_ids=params['obj_ids'],
                       label_mapping=params['id_2_trainid'])
        label_gt_nd = mx.nd.array(label_gt)

        pred = mx.nd.array(np.expand_dims(
                  np.transpose(label_prob, [2, 0, 1]), axis=0))
        eval_metric.update([label_gt_nd], [pred])

        logging.info('Eval {}'.format(eval_metric.get()))
        cost_time = time.time() - start_time
        time_all += cost_time

        print "\t infer time cost {}".format(cost_time)
        label = uts.prob2label(label_prob)
        if args.is_save:
            is_write = cv2.imwrite(save_path + image_name + '.png', label)
            if not is_write:
                print save_path + image_name
                raise ValueError('not able to write')

        # bkg_color = [255, 255, 255]
        # label_gt_c = uts.label2color(label_gt[0,:,:],
        #         params['color_map_list'],
        #         bkg_color)
        # label_c = uts.label2color(label, params['color_map_list'],
        #                           bkg_color)
        # uts.plot_images({'image': image,
        #                  'label_pred': label_c,
        #                  'label_gt': label_gt_c})

    name, res = eval_metric.get()
    logging.info('Final eval {} ave time {}'.format(str((name, res)), time_all / len(image_list)))

    ious = eval_metric.get_ious()
    pixel_num = eval_metric.get_pixel_num()
    for i, class_name in enumerate(params['class_names']):
        if not np.isnan(ious[i]):
            print '%s: %05f, %d \t' % (class_name, ious[i], pixel_num[i])

    res = res + ious.tolist()
    np.savetxt(save_path + 'res.txt', np.array(res))


def test_pose_seg(is_refine):
    import MXlayers.init_proj as init_proj
    suppress = 1

    ctx = mx.gpu(int(args.gpu_ids))
    params = zpark.set_params()
    data_config, label_config = data_iter.get_pose_seg_setting()
    height, width = data_config['image']['size']
    omit_classes = [i for i in range(params['class_num']) if \
            (not params['is_exist'][i]) or params['is_obj'][i] \
             or params['is_unknown']]

    data_names = ['image', 'pose_in']
    data_shapes = [tuple([1, 3] + data_config['image']['size']) ]
    data_shapes += [tuple([1, 6])]
    params['batch_size'] = 1

    init_proj.init(params, data_config, [height, width])
    trans_ext = 'Loc/'
    augmentor = None

    if args.is_augment:
        augmentor = aug.TestAugmentor(['flip']) # set default parameters

    net, out_names = def_model_pose_segnet(data_names,params,
            args.net_name, suppress=suppress)
    models = {'model': args.test_model}
    model = nuts.load_model(models, data_names, data_shapes, net, ctx)
    method = args.test_model.split('/')[-1]
    image_list = [line.strip() for line in open(params['test_set'])]
    eval_metric = metric.SegMetric(ignore_label=[0] + omit_classes)

    time_all = 0.
    for count, image_name in enumerate(image_list):
        image_path = params['image_path'] + image_name + '.jpg'
        trans_path = params['output_path'] + trans_ext + image_name + '.txt'
        label_path = params['label_path'] + image_name + '.png'
        save_path = params['output_path'] + method + '/'

        scene = '/'.join(image_name.split('/')[:-1])
        uts.mkdir_if_need(save_path + scene)
        if os.path.exists(save_path + image_name + '.png'):
            continue

        print '\t segmenting image {}'.format(image_name),
        image = cv2.imread(image_path)
        _, pose_in, _, _ = data_config['pose_in']['reader'](
                trans_path,
                **data_config['pose_in']['reader_params'])

        inputs = {'image':image, 'pose_in':pose_in}
        label_gt = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        start_time = time.time()
        label_prob = infer(model, inputs, data_config,
                           out_names,
                           augmentor=augmentor,
                           eval_metric = eval_metric,
                           label=label_gt,
                           label_setting=label_config)

        cost_time = time.time() - start_time
        time_all += cost_time
        print "\t infer time cost {}".format(cost_time)
        label = uts.prob2label(label_prob[out_names[-1]])

        if args.is_save:
            is_write = cv2.imwrite(save_path + image_name + '.png', label)
            if not is_write:
                raise ValueError('not able to write')

        bkg_color = [255, 255, 255]
        label_c = uts.label2color(label, params['color_map_list'],
                                  bkg_color)
        uts.plot_images({'image':image,
                         'label_pred': label_c})

    logging.info('Final eval {} ave time {}'.format(str(eval_metric.get()), time_all / len(image_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the trained model for acc.')
    parser.add_argument('--test_model', default='./Output/rnn-3d-refine-0200',
            help='The prefix(include path) of vgg16 model with mxnet format.')
    parser.add_argument('--net_name', default='segment', help='The segmentation name')
    parser.add_argument('--gpu_ids', default="1", help='the gpu ids use for training')
    parser.add_argument('--has_3d', action='store_true', default=False,
        help='whether to save the results')
    parser.add_argument('--is_rnn', action='store_true', default=False,
        help='whether to save the results')
    parser.add_argument('--pre_render', action='store_true',
        help='whether have pre rendered resutls')
    parser.add_argument('--is_save', action='store_true',
        help='whether to save the results')
    parser.add_argument('--is_refine', action='store_true', default=True,
        help='whether to visualize the results')
    parser.add_argument('--is_augment', action='store_true',
        help='whether to augment the results')

    args = parser.parse_args()
    logging.info(args)
    method = 'posenet-proj-v2-0074'

    # pdb.set_trace()
    # test_seg(args.is_refine, 'gt', 'bkgobj')
    # test_seg(args.is_refine, 'gt', 'binary')
    if not args.has_3d:
        print 'test image only'
        test_seg(args.is_refine, None, 'full')
    else:
        print 'test with gt'
        test_seg(args.is_refine, 'gt', 'full')

    # is_rnn=False
    # test_seg_multi(args.is_refine, 'posenet-sem-wei-proj-0010', 'bkgobj', False, 'zpark')
    # test_seg_multi(args.is_refine, 'posenet_rnn_hier_euler-0061', 'bkgobj', True, 'zpark')

    # test_seg_multi(args.is_refine, 'pose-rnn-0120', 'full', args.is_rnn, 'dlake')
    # test_seg(args.is_refine, '', 'bkgobj')

    # test_sim_data()
    # test_pose_seg(args.is_refine)


