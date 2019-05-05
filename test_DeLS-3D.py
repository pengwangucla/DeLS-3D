"""
Test demo for pretrained models.
Author: 'peng wang'

"""

import argparse
import cv2
import sys
import logging
import mxnet as mx
import numpy as np
import time
import vis_utils as uts
import vis_metrics as metric
import data_transform as ts
import utils_3d as uts_3d
import utils.dels_utils as de_uts

from dataset import data_lib

import networks.pose_nn as pose_nn
import networks.seg_nn as seg_nn
import networks.net_util as nuts
from config import config
from collections import namedtuple, OrderedDict
debug = 0

np.set_printoptions(precision=4, suppress=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

Batch = namedtuple('Batch', ['data'])
input_type = metric.InputType

def infer(mod,
          inputs,
          data_setting,
          out_names):

    assert isinstance(inputs, OrderedDict)
    data_list = []
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

    mod.forward(Batch(data_list))
    output_nd = mod.get_outputs()

    res = {}
    for i, name in enumerate(out_names):
        output_nd[i].wait_to_read()
        res[name] = output_nd[i].asnumpy()

    return res


class DeLS3D(object):
    def __init__(self,
                 dataname,
                 pose_cnn=None,
                 pose_rnn=None,
                 seg_cnn=None,
                 noisy_pose_path=None,
                 is_train=False):
        """
        Performing deep localization and segmentation on validation set
        of corresponding data.
            Only support batch size 1 for simplicity.
            (TODO) add training code

        """
        # targeting at specific scenes with 3D point cloud
        assert dataname in ['zpark']
        # (TODO)  'dlake' part is not tested yet

        if not is_train:
            assert pose_cnn and pose_rnn and seg_cnn

        self.dataname = dataname
        self.params = data_lib[dataname].set_params()

        # we merge multiple object class to single for zpark
        # due to few instances
        self.seg_gt_type = {'zpark': 'bkgobj',
                            'dlake': 'full'}
        print(self.params['size'])
        self.renderer = de_uts.Renderer(self.params,
                                        self.params['cloud'],
                                        self.params['size'])
        self.models = {'pose_cnn': pose_cnn,
                       'pose_rnn': pose_rnn,
                       'seg_cnn': seg_cnn}

        # whether use quaternion representation for pose
        self.is_quater = False
        self.pose_size = 7 if self.is_quater else 6
        self.sim_noisy_params = {}
        self.ctx = mx.gpu(int(args.gpu_ids))
        self.test_image_list = [line.strip() for line in \
                                open(self.params['test_set'])]
        self.image_num = len(self.test_image_list)

        # define the path for refinement
        self.noisy_pose_path = self.params['noisy_pose_path'] \
                if noisy_pose_path is None else noisy_pose_path
        self.pose_cnn_output_path = '%s/pose_cnn/' % \
                                    self.params['output_path']
        self.pose_rnn_output_path = '%s/pose_rnn/' % \
                                    self.params['output_path']
        self.seg_cnn_output_path = '%s/seg_cnn/' % \
                                    self.params['output_path']


    def def_model(self, in_names, network='pose_cnn',
                  is_train=False, **args):
        """
           set up the symbolic of different networks for inference
        """

        inputs = nuts.get_mx_var_by_name(in_names)
        if network == 'pose_cnn':
            net = pose_nn.pose_block(inputs,
                        self.params,
                        in_names=in_names,
                        name='pose',
                        is_train=is_train)

        elif network == 'pose_rnn':
            net = pose_nn.recurrent_pose(inputs, name='pose',
                        is_train=is_train, **args)

        elif network == 'seg_cnn':
            net_name = 'segment'
            block = seg_nn.segment_block(inputs,
                                         self.params,
                                         name=net_name,
                                         is_train=False)
            arg_params = nuts.def_arguments(
                           block[net_name + '_score'].list_arguments())

            net = seg_nn.recurrent_seg_block(inputs,
                                             self.params,
                                             name=net_name,
                                             in_names=in_names,
                                             is_train=False,
                                             spatial_steps=1,
                                             arg_params=arg_params,
                                             **args)
        else:
            raise ValueError('no such network')

        return mx.symbol.Group(net.values()), net.keys()


    def get_sim_seq(self, re_generate=True):
        """ Use a sample sequence pregenerated
        """
        if re_generate:
            logging.info('regenerate simulation seq in %s' % \
                         self.noisy_pose_path)
            for i, image_name in enumerate(self.test_image_list):
                pose_gt_file = '%s/%s.txt' % (self.params['pose_path'],
                        image_name)
                pose_gt = de_uts.loadtxt(pose_gt_file)
                trans_i, rot_i = uts_3d.random_perturb(pose_gt[:3],
                        pose_gt[3:], 5.5, 12.0 * np.pi / 180.0)
                noisy_pose = np.concatenate([trans_i, rot_i])
                noisy_pose_file = '%s/%s.txt' % (self.noisy_pose_path,
                        image_name)
                np.savetxt(noisy_pose_file, noisy_pose)

            # necessary for pose cnn
            self.render_segments(self.noisy_pose_path)

        # self.eval_pose(self.noisy_pose_path)
        return self.params['noisy_pose_path']


    def localize_and_segment(self,
                             noisy_pose_path=None,
                             render_seg_for_pose=True,
                             is_eval=True):
        """
           self.pose_seq: sequence of noisy GPS/IMU
           one may also adjust the
        """
        if noisy_pose_path:
            self.noisy_pose_path = noisy_pose_path

        if is_eval:
            self.eval_pose(self.noisy_pose_path, False)

        self.refine_with_pose_cnn()
        if is_eval:
            self.eval_pose(self.pose_cnn_output_path, False)

        self.refine_with_pose_rnn()
        self.render_segments(self.pose_rnn_output_path)
        if is_eval:
            self.eval_pose(self.pose_rnn_output_path, True)

        self.get_segment(self.pose_rnn_output_path)
        if is_eval:
            self.eval_segments(self.seg_cnn_output_path,
                               self.seg_gt_type[self.dataname])

        uts.frame_to_video(image_path=self.params['image_path'],
                            label_path=self.seg_cnn_output_path,
                            frame_list=self.test_image_list,
                            color_map=self.params['color_map_list'],
                            video_name='%s/video.avi' % self.seg_cnn_output_path)


    def refine_with_pose_cnn(self):
        # first stage do perimage refining
        # define saving place
        data_setting, label_setting = config.network.pose_cnn_setting(
                                        with_pose_in=True)
        height, width = data_setting['image']['size']

        # init model
        data_names = ['image', 'label_db', 'pose_in']
        data_shapes = [tuple([1, 3] + data_setting['image']['size']),
                       tuple([1, 1] + data_setting['label_db']['size']),
                       tuple([1, 6])]

        net, out_names = self.def_model(data_names, version='cnn')
        model = nuts.load_model({'pose_cnn': self.models['pose_cnn']},
                                data_names=data_names,
                                data_shapes=data_shapes,
                                net=net, ctx=self.ctx)

        for i, image_name in enumerate(self.test_image_list):
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
            time_s = time.time()
            res = infer(model, inputs, data_setting, out_names)
            pose_out = res[out_names[0]]

            self._get_save_path(self.pose_cnn_output_path, image_name)
            np.savetxt('%s/%s.txt' % (self.pose_cnn_output_path, image_name),
                       pose_out)
            time_cost = time.time() - time_s
            sys.stdout.write('\r>> refine %s with pose_cnn %d/%d, time: %04f'\
                     % (image_name, i, len(self.test_image_list), time_cost))
            sys.stdout.flush()


    def _get_pose_rnn_mutable_module(self, image_list, arg_params, aux_params):
        """
           For testing trained pose rnn model, for training we always use a 100
         length sequence
        """

        image_num = len(image_list)
        data_names = ['pose_in_%03d' % i for i in range(image_num)]
        data_shapes = [tuple([1, self.pose_size]) \
                                         for i in range(image_num)]
        data_shapes = [(name, shape) for name, shape \
                                   in zip(data_names, data_shapes)]

        inputs = nuts.get_mx_var_by_name(data_names)
        net = pose_nn.recurrent_pose(inputs,
                name='pose',
                out_num=self.pose_size,
                is_train=False,
                is_highorder=True,
                layer_num=2)

        net = mx.sym.Group(net.values())
        mod = mx.mod.Module(net,
                            data_names=data_names,
                            label_names=None,
                            context=self.ctx)
        mod.bind(for_training=False, data_shapes=data_shapes)
        mod.set_params(arg_params,
                       aux_params,
                       allow_missing=False)
        return mod

    def refine_with_pose_rnn(self, render_segment=False):
        """refine the output from pose_cnn using pose_rnn
        """
        image_list_all = [line.strip() for line in open(self.params['test_set'])]
        image_seq = de_uts.image_set_2_seqs(image_list_all,
                                               self.params['cam_names'])
        logging.info('test seqs len %d' % len(image_seq))
        rnn_model = {'pose_rnn':self.models['pose_rnn']}
        _, arg_params, aux_params = nuts.args_from_models(rnn_model, False)

        for count, image_list in enumerate(image_seq):
            logging.info('predict sequence %d/%d, len %d' % (
                count, len(image_seq), len(image_list)))

            # read
            mod = self._get_pose_rnn_mutable_module(image_list,
                                                    arg_params,
                                                    aux_params)

            pose_in = []
            for image_name in image_list:
                pose_file = '%s/%s.txt' % (self.pose_cnn_output_path,
                         image_name)
                pose_in_cur = de_uts.loadtxt(pose_file)
                if self.is_quater:
                    pose_in_cur = np.concatenate([pose_in_cur[:3],
                            uts_3d.euler_angles_to_quaternions(
                                    pose_in_cur[3:6])])
                pose_in_cur = ts.pose_transform(pose_in_cur)
                pose_in.append(mx.nd.array(pose_in_cur))

            mod.forward(Batch(pose_in))
            output_nd = mod.get_outputs()

            for res, image_name in zip(output_nd, image_list):
                self._get_save_path(self.pose_rnn_output_path, image_name)
                pose_out = res.asnumpy()[0]
                if self.is_quater:
                    pose_out[3:] = pose_out[3:]/np.linalg.norm(pose_out[3:])
                    pose_out = np.concatenate([pose_out[:3],
                            uts_3d.quaternions_to_euler_angles(pose_out[3:6])])

                # save results
                np.savetxt('%s/%s.txt' % (self.pose_rnn_output_path,
                           image_name), pose_out)

        if render_segment:
            self.render_segments(self.pose_rnn_output_path)

    def get_segment(self, pose_path=None):
        """
           Get segmentation results with localization
        """
        if not pose_path:
            pose_path = self.pose_rnn_output_path

        # setting the experiment
        data_config, label_config = config.network.seg_cnn_setting(True)
        height, width = data_config['data']['size']

        self.params['batch_size'] = 1
        data_names = ['data', 'label_db']
        data_shapes = [tuple([1, data_config[name]['channel']] \
                        + data_config[name]['size']) for name in data_names]

        model_args = {'is_refine': True, 'suppress':1}
        net, out_names = self.def_model(in_names=data_names,
                                        network='seg_cnn', **model_args)

        models = {'seg_cnn': self.models['seg_cnn']}
        model = nuts.load_model(models, data_names, data_shapes, net, self.ctx)

        for i, image_name in enumerate(self.test_image_list):
            image_path = self.params['image_path'] + image_name + '.jpg'
            label_db_file = '%s/%s.png' % (pose_path, image_name)

            # load the predicted pose
            start_time = time.time()
            image = de_uts.imread(image_path)
            label_db = de_uts.imread(label_db_file)

            inputs = OrderedDict([('data', image),
                                  ('label_db', label_db)])
            label_prob = infer(model, inputs,
                               data_config,
                               out_names)
            label_prob = label_prob[out_names[-1]]
            label_prob = np.transpose(label_prob[0], (1, 2, 0))
            label = uts.prob2label(label_prob)
            time_cost = time.time() - start_time

            self._get_save_path(self.seg_cnn_output_path, image_name)
            is_write = cv2.imwrite('%s/%s.png' % (self.seg_cnn_output_path, image_name), label)
            if not is_write:
                raise ValueError('not able to write')
            self.counter(image_name, i, time_cost, name='seg_cnn')

    def _vis_seg_cnn_comparison(self, image_name):
        """ Visualization of segmentation results
        """
        image_file = '%s/%s.jpg' % (self.params['image_path'], image_name)
        image = de_uts.imread(image_file)

        bkg_color = [255, 255, 255]
        image_file = '%s/%s.png' % (self.seg_cnn_output_path,
                                    image_name)
        label_res = de_uts.imread(image_file)
        label_c = uts.label2color(label_res,
                                  self.params['color_map_list'],
                                  bkg_color)

        image_file = '%s/%s.png' % (self.params['semantic_label'],
                image_name)
        label_gt = de_uts.imread(image_file)
        label_gt_c = uts.label2color(label_gt,
                         self.params['color_map_list'], bkg_color)

        image_file = '%s/%s.png' % (self.pose_rnn_output_path,
                                    image_name)
        label_db = de_uts.imread(image_file)
        label_db_c = uts.label2color(label_db,
                self.params['color_map_list'], bkg_color)
        uts.plot_images({'image':image,
                         'label_db': label_db_c,
                         'label_pred': label_c,
                         'label_gt': label_gt_c})


    def _get_save_path(self, output_path, image_name):
        """ get the corresponding saving path from image name in apolloscape
            dataset
        """

        test_scene = '/'.join(image_name.split('/')[:-1])
        save_path = '%s/%s/' % (output_path, test_scene)
        uts.mkdir_if_need(save_path)

        return save_path

    def counter(self, image_name,  i, time_cost=0, name='func'):
        sys.stdout.write('\r>> processing %s with %s %d/%d, time: %04f'\
                 % (image_name, name, i, len(self.test_image_list), time_cost))
        sys.stdout.flush()


    def render_segments(self, pose_path):
        """Render labelled image out from generated poses
        """
        for i, image_name in enumerate(self.test_image_list):
            pose_file = '%s/%s.txt' % (pose_path, image_name)
            pose_in = de_uts.loadtxt(pose_file)
            intr_key = de_uts.get_intr_key(self.params['intrinsic'].keys(),
                                              image_name)
            s = time.time()
            label_db = self.renderer.render_from_3d(pose_in,
                                         self.params['intrinsic'][intr_key],
                                         self.params['color_map'],
                                         is_color=self.params['is_color_render'],
                                         label_map=self.params['id_2_trainid'])
            if debug:
                bkg_color = [255, 255, 255]
                label_db_c = uts.label2color(label_db,
                                             self.params['color_map_list'],
                                             bkg_color)
                label_db_prev = de_uts.imread('%s/%s.png' % (pose_path, image_name))
                label_db_prev_c = uts.label2color(label_db_prev,
                                             self.params['color_map_list'],
                                             bkg_color)

                uts.plot_images({'label_db': label_db_c, 'label_db_prev': label_db_prev_c}, layout=(1, 2))

            time_cost = time.time() - s
            self._get_save_path(pose_path, image_name)
            cv2.imwrite('%s/%s.png' % (pose_path, image_name), label_db)

            self.counter(image_name, i, time_cost, name='render image')


    def get_refined_pose_path(self):
        return self.pose_cnn_output_path


    def get_refined_segment_path(self):
        return self.seg_cnn_output_path


    def eval_pose(self, pose_res_path, eval_render_segment=False):
        """
        with_segment: if also evaluate projected segments
        """
        # start eval
        import vis_metrics as metric
        pose_metric = metric.get_pose_metric(input_type.NUMPY)(is_euler=True)
        pose_metric.reset()

        logging.info('Eval Path {}'.format(pose_res_path))
        for i, image_name in enumerate(self.test_image_list):
            gt_file = self.params['pose_path'] + image_name + '.txt'
            pose_gt = de_uts.loadtxt(gt_file)
            res_file = '%s/%s.txt' % (pose_res_path, image_name)
            pose_out = de_uts.loadtxt(res_file)
            pose_metric.update([pose_gt[None, :]],
                               [pose_out[None, :]])
            self.counter(image_name, i, 0, name='eval pose')

        logging.info('Pose Error {}'.format(str(pose_metric.get())))

        if eval_render_segment:
            self.eval_segments(seg_res_path=pose_res_path, gt_type='bkg')


    def eval_segments(self, seg_res_path, gt_type):
        """
          evaluate segmentation with metric

          gt_type:
              this is noly for zpark dataset, for dlake dataset, we haven't done
              background label inpainted.

            'bkg': rendered bkg map
            'bkgfull': inpainted bkg map
            'bkgobj': inpainted bkg map + object mask
            'full': inpainted bkg map + separated object mask
        """

        def get_gt_label_path(gt_type):
            """
               Obtain corresponding gt label for evaluation
            """
            if gt_type in ['bkg', 'bkgfull']:
                label_path = 'label_%s_path' % gt_type
            elif gt_type :
                label_path = 'label_path'

            # useful for zpark since we remap label
            label_mapping = data_lib[self.dataname].get_label_mapping(
                    gt_type, self.params)
            return self.params[label_path], label_mapping

        ignore_labels = [0, 255]
        seg_gt_path, label_mapping = get_gt_label_path(gt_type)
        seg_metric = metric.get_seg_metric(input_type.MXNET)(
                                           ignore_label=ignore_labels)
        seg_metric.reset()
        for i, image_name in enumerate(self.test_image_list):
            # transform segment results
            res_file = '%s/%s.png' % (seg_res_path, image_name)
            label_res = de_uts.imread(res_file)
            seg_output = mx.nd.array(label_res[None, :, :])
            seg_output = mx.nd.one_hot(seg_output,
                                       self.params['class_num'])
            seg_output = mx.nd.transpose(seg_output, axes=(0, 3, 1, 2))

            # transform segment ground truth
            height, width = label_res.shape[:2]
            gt_file = '%s/%s.png' % (seg_gt_path, image_name)
            label_gt = de_uts.imread(gt_file)
            label_gt = cv2.resize(label_gt, (width, height),
                                  interpolation=cv2.INTER_NEAREST)
            label_seg = mx.nd.array(ts.label_transform(label_gt,
                                    label_mapping=label_mapping))
            time_s = time.time()
            seg_metric.update([label_seg], [seg_output])
            time_cost = time.time() - time_s
            self.counter(image_name, i, time_cost, name='eval segment')

        logging.info('\n Segment Accuracy {}'.format(str(seg_metric.get())))



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

    # then load the trained networks cnn rnn seg_cnn,
    DeLS = DeLS3D(args.data,
                  args.pose_cnn,
                  args.pose_rnn,
                  args.seg_cnn)

    # first simulate a perturbation sequence using args.dataset
    # pose_seq = uts.simulate_permuted_sequence(
    #                              args.dataset, split='val')

    # get the sequence pre-simulated
    save_path = DeLS.get_sim_seq(re_generate=False)
    DeLS.localize_and_segment(noisy_pose_path=save_path)





