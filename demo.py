# to be implemented concatenate pose network and segment cnn

import argparse
import cv2
import logging
import mxnet as mx
import numpy as np
import metric
import time
import vis_utils.utils as uts
import vis_utils.utils_3d as uts_3d

import config
import dataset.data_iters as data_iter
import networks.pose_nn as pose_nn
import networks.net_util as nuts
from collections import namedtuple, OrderedDict
import pdb

def infer(mod, inputs,
          data_setting,
          out_names):

    assert isinstance(inputs, OrderedDict)
    data_list = []
    s = time.time()
    for in_name in inputs:
        data = np.array(inputs[in_name], dtype=np.float32)
        # (h, w, c)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the trained model for acc.')
    parser.add_argument('--network', default='',
            help='The model name with mxnet format.')
    parser.add_argument('--dataset', default='dlake',
            help='The dataset name with mxnet.')
    parser.add_argument('--gpu_ids', default="1",
            help='the gpu ids use for training')
    parser.add_argument('--is_save', default="1",
            help='whether to save the rendered results')

    args = parser.parse_args()
    logging.info(args)

    test_posenet()

