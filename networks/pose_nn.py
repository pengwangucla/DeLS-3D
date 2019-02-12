"""Pose network for 3D pose denoise
"""

import mxnet as mx
from collections import OrderedDict
import util_layers as utl
import net_util as net_util
from base_models import googlenet


def pose_net_loss(rot, trans, label, balance=(1.0, 1.0)):
    trans_gt = mx.sym.slice_axis(label, axis=1, begin=0, end=3)
    rot_gt = mx.sym.slice_axis(label, axis=1, begin=3, end=7)

    rot = mx.symbol.L2Normalization(rot, mode='instance')
    rot_diff = rot - rot_gt

    trans_diff = trans - trans_gt
    trans_loss = mx.symbol.smooth_l1(data=trans_diff, scalar=1)
    trans_loss = mx.symbol.sum(trans_loss, axis=1)
    rot_loss = mx.symbol.smooth_l1(data=rot_diff, scalar=1)
    rot_loss = mx.symbol.sum(rot_loss, axis=1)
    total_loss = rot_loss * balance[0] + trans_loss * balance[1]
    return total_loss


def pose_loss(pose, label, balance=1000, loss_type='xyz_e'):
    trans_gt = mx.sym.slice_axis(label, axis=1, begin=0, end=3)
    trans = mx.sym.slice_axis(pose, axis=1, begin=0, end=3)

    if 'xyz_q' == loss_type:
        rot = mx.sym.slice_axis(pose, axis=1, begin=3, end=7)
        rot_gt = mx.sym.slice_axis(label, axis=1, begin=3, end=7)
        rot = mx.sym.L2Normalization(rot, mode='instance')

    elif 'xyz_e' == loss_type:
        rot = mx.sym.slice_axis(pose, axis=1, begin=3, end=6)
        rot_gt = mx.sym.slice_axis(label, axis=1, begin=3, end=6)

    trans_diff = trans - trans_gt
    rot_diff = rot - rot_gt
    trans_loss = mx.sym.smooth_l1(data=trans_diff, scalar=1)
    rot_loss = mx.sym.smooth_l1(data=rot_diff, scalar=1)
    trans_loss = mx.sym.sum(trans_loss, axis=1)
    rot_loss = mx.sym.sum(rot_loss, axis=1)

    total_loss = rot_loss * balance + trans_loss

    return total_loss


def projective_loss(pose, label, point_3d, weight=None):
    Rt_pose = net_util.pose2mat(pose)
    Rt_label = net_util.pose2mat(label)

    res = net_util.world2cam(point_3d, Rt_pose)
    res_gt = net_util.world2cam(point_3d, Rt_label)

    res_diff = res - res_gt
    proj_loss = mx.symbol.smooth_l1(data=res_diff, scalar=1)
    proj_loss = mx.symbol.sum(proj_loss, axis=1)
    if weight is not None:
        proj_loss = proj_loss * weight

    proj_loss = mx.symbol.sum(proj_loss, axis=1)

    return proj_loss


def gen_pose_loss(pose, label,
                  ext_inputs=None,
                  loss_types=['xyz_e']):
    """Different pose losses

    Return:
        A list of pose loss for final usage
    """
    loss_all = []
    for loss in loss_types:
        if loss == 'xyz_e' or loss == 'xyz_q':
            cur_loss = pose_loss(pose, label, loss_type=loss)
            cur_loss = mx.sym.MakeLoss(data=cur_loss, grad_scale=1.0)
            loss_all.append(cur_loss)

        elif 'proj' == loss:
            point_3d = ext_inputs['points']
            weight = ext_inputs['mask']
            point_3d = mx.sym.stop_gradient(point_3d)
            cur_loss = projective_loss(pose, label, point_3d, weight)
            cur_loss = mx.symbol.MakeLoss(data=cur_loss, grad_scale=1.0)
            loss_all.append(cur_loss)
        else:
            raise ValueError('No such loss type %s' % loss)

    return loss_all


def pose_block(inputs,
               data_params,
               in_names=['image', 'label_db', 'pose_in'],
               name='pose',
               ext='',
               ext_inputs=None,
               iter_num=None,
               arg_params=None,
               is_train=True,
               label=None,
               loss_type='proj'):

    iter_name = '' if iter_num is None else '_' + str(iter_num)
    iter_name = iter_name + ext

    fc_bn_layer = utl.FCBNLayer(ext=iter_name, params=arg_params)
    conv_bn_layer = utl.ConvBNLayer(ext=iter_name, params=arg_params)

    image = inputs[in_names[0]]
    label_map = inputs[in_names[1]] / 32
    in_source = mx.symbol.concat(image, label_map, dim=1)

    conv_feat = utl.conv_block(in_source,
            arg_params=arg_params, name=name, ext=iter_name)
    conv_feat = conv_feat['block5_1']

    motion_feat = conv_bn_layer(conv_feat, 128, 3, 1, name=name + '_motion')
    fc1 = fc_bn_layer(motion_feat, 1024, name=name + '_fc1')
    fc2 = fc_bn_layer(fc1, 128, name=name + '_fc2')
    pose = fc_bn_layer(fc2, 6, act=None, name=name + '_pose')
    pose = pose + inputs[in_names[2]]

    # print 'test only forward'
    # pose = mx.sym.stop_gradient(pose)
    Outputs = OrderedDict([(name + '_output' + iter_name, pose)])
    if is_train:
        loss_all = []
        if 'xyz_e' in loss_type:
            cur_loss = pose_loss(pose, label)
            cur_loss = mx.sym.MakeLoss(data=cur_loss, grad_scale=1.0)
            loss_all.append(cur_loss)

        elif 'proj' in loss_type:
            point_3d = ext_inputs['points']
            weight = ext_inputs['weight']
            point_3d = mx.sym.stop_gradient(point_3d)
            weight = mx.sym.stop_gradient(weight)
            cur_loss = projective_loss(pose, label, point_3d, weight)
            cur_loss = mx.symbol.MakeLoss(data=cur_loss, grad_scale=1.0)
            loss_all.append(cur_loss)

        loss_all.append(mx.sym.BlockGrad(pose, name='pose'))
        Outputs[name + '_loss' + iter_name] = mx.symbol.Group(loss_all)

    return Outputs


def pose_subnet(feat, pool_size=5, stride=3, filter_size=1024,
        drop_rate=0.7, pose_in=None, name='cls', is_last=False):
    fc_bn_layer = utl.FCBNLayer()
    conv_bn_layer = utl.ConvBNLayer()

    kernel = (pool_size, pool_size)
    stride = (stride, stride)
    pool = mx.symbol.Pooling(feat, kernel=kernel, stride=stride, pool_type='avg')

    if not is_last:
        conv1 = conv_bn_layer(pool, 128, 1, 1, act='RELU', name=name+'_reduction_pose')
    else:
        conv1 = pool

    ext = ''
    if pose_in:
        pose_in_feat = fc_bn_layer(pose_in, 128, act='RELU', name=name+'_pose_in_feat')
        conv1 = mx.symbol.concat(conv1, pose_in_feat, dim=1)
        ext = '_ext_feat'

    fc1 = fc_bn_layer(conv1, filter_size, act='RELU', name=name+'_fc1_pose'+ext)
    fc1 = mx.symbol.Dropout(fc1, p=drop_rate, name=name+'_dropout')

    trans = fc_bn_layer(fc1, 3, act='RELU', name=name+'_fc_pose_xyz'+ext)
    rot = fc_bn_layer(fc1, 4, act='RELU', name=name+'_fc_pose_wpqr'+ext)

    return (trans, rot)


def pose_net(image,
            name='pose',
            ext='',
            pose_in=None,
            arg_params=None,
            is_train=True,
            label=None):
    if pose_in:
        trans_in = mx.sym.slice_axis(pose_in, axis=1, begin=0, end=3)
        rot_in = mx.sym.slice_axis(pose_in, axis=1, begin=3, end=7)
    else:
        trans_in = 0
        rot_in = 0

    feature = googlenet(image)
    feat_names = ['icp3_out', 'icp6_out', 'icp9_out']
    motion_pred = OrderedDict({})
    motion_pred[feat_names[0]] = pose_subnet(feature[feat_names[0]], pose_in=pose_in,
            name='cls1')
    motion_pred[feat_names[1]] = pose_subnet(feature[feat_names[1]], pose_in=pose_in,name='cls2')
    motion_pred[feat_names[2]] = pose_subnet(feature[feat_names[2]], pool_size=7,
            stride=1, filter_size=2048, drop_rate=0.5, pose_in=pose_in, name='cls3', is_last=True)
    xyz = motion_pred[feat_names[2]][0] + trans_in
    wpqr = motion_pred[feat_names[2]][1] + rot_in
    wpqr = mx.symbol.L2Normalization(wpqr, mode='instance')
    Outputs = OrderedDict({'xyz': xyz, 'wpqr': wpqr})

    if is_train:
        pose_losses = []
        for feat_name in feat_names:
            this_trans = motion_pred[feat_name][0] + trans_in
            this_rot = motion_pred[feat_name][1] + rot_in
            this_rot = mx.symbol.L2Normalization(this_rot, mode='instance')
            pose_loss = pose_net_loss(this_rot, this_trans, label)
            pose_losses.append(mx.symbol.MakeLoss(pose_loss))

        final_pose = mx.sym.concat(motion_pred[feat_names[-1]][0],
                motion_pred[feat_names[-1]][1], dim=1)
        pose_losses.append(mx.sym.BlockGrad(final_pose, name='pose'))
        loss_all = mx.symbol.Group(pose_losses)
        Outputs[name + '_loss'] = loss_all
    return Outputs



def recurrent_pose(inputs, name,
        dep_num=2,
        layer_num=1,
        out_num=6,
        data_params=None,
        ext_inputs=None,
        is_highorder=False,
        recurrent_cell='GRU',
        is_train=True,
        labels=None,
        loss_types=['xyz_q']):

    """Simple RNN layer for refine pose predictions
    Inputs:
       inputs is a list of pose
       labels is a list of labels
       dep_num: dependent frame
       layer_num: how many layer for rnn
       out_num: how many dimension
       loss_types: optional, list of loss ['xyz_e', 'xyz_q', 'proj']
    Output:
       Loss or rectified pose
    """

    Outputs = OrderedDict({})
    # define shared paramters
    arg_params_name = []
    for suffix in ['_weight', '_bias']:
        arg_params_name = arg_params_name + [name + '_embed' + suffix,
                name + '_pred' + suffix]
    arg_params = net_util.get_mx_var_by_name(arg_params_name)
    fc_bn_layer = utl.FCBNLayer(params=arg_params)

    in_num = len(inputs)
    state_num = 32

    inputs_new = []
    if not is_highorder:
        inputs_new = inputs.values()
    else:
        # try to have higher order relationship
        keys = inputs.keys()
        dep_order = range(dep_num)[::-1] if layer_num == 1 \
                                         else range(dep_num)
        for i in range(in_num):
            sub_seq = []
            for j in dep_order:
                last = max(i - j, 0)
                sub_seq.append(inputs[keys[last]])
            inputs_new.append(mx.sym.concat(*(sub_seq), dim=1))

    # first needs to embedding
    for i in range(in_num):
        inputs_new[i] = fc_bn_layer(
                inputs_new[i], state_num, act=None,
                name=name+'_embed')

    # then rnn cell for hidden state
    if 'MLP' == recurrent_cell:
        feat = [mx.sym.LeakyReLU(data=inputs_new[i], act_type='rrelu') \
                for i in range(in_num)]

    elif 'GRU' == recurrent_cell:
        cell = mx.rnn.ResidualCell(mx.rnn.GRUCell(state_num, prefix=name))
        feat, _ = cell.unroll(in_num, inputs_new)

        # add another layer
        if layer_num == 2:
            cell_2 = mx.rnn.ResidualCell(mx.rnn.GRUCell(
                state_num, prefix=name + '_2'))
            feat, _ = cell_2.unroll(in_num, feat)

    # finally prediction layer to predict the residual
    pose = []
    for i in range(in_num):
        pose.append(fc_bn_layer(feat[i], out_num, act=None,
            name=name+'_pred'))

    pose = mx.sym.concat(*pose, dim=0)
    pose_in = mx.sym.concat(*(inputs.values()), dim=0)
    pose = pose + pose_in

    if is_train:
        label = mx.sym.concat(*(labels.values()), dim=0)
        poses = mx.sym.split(pose, axis=0, num_outputs=in_num)

        proj_loss = 'proj' in loss_types
        if proj_loss:
            points = mx.sym.concat(*(ext_inputs.values()), dim=0)

        pose_loss = gen_pose_loss(pose, label,
                ext_inputs={'points':points} if proj_loss else None,
                loss_types=loss_types)

        pose_loss = pose_loss + [mx.sym.BlockGrad(poses[i],
            name='pose_%02d' % i) for i in range(in_num)]
        Outputs[name + '_loss'] = mx.symbol.Group(pose_loss)
        return Outputs
    else:
        poses = mx.sym.split(pose, axis=0, num_outputs=in_num)
        for i in range(in_num):
            Outputs['pose_out_%03d' % i] = poses[i]
        return Outputs


if __name__=='__main__':
    # data_names = ['image', 'pose_in']
    # label_names = ['pose']
    # input = net_util.get_mx_var_by_name(data_names)
    # label = net_util.get_mx_var_by_name(label_names)
    # net_out = pose_net(input['image'], pose_in=input['pose_in'],
    #         is_train=True,label=label['pose'])
    # mx.viz.print_summary(net_out['pose_loss'])
    data_names = ['pose_in_00', 'pose_in_01']
    label_names = ['pose_00', 'pose_01']
    inputs = net_util.get_mx_var_by_name(data_names)
    labels = net_util.get_mx_var_by_name(label_names)
    net_out = recurrent_pose(inputs, name='pose_', is_train=False,
            is_highorder=True)
    # mx.viz.print_summary(net_out['pose_out_001'])
    mx.viz.plot_network(net_out['pose_out_001'])




