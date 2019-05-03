import mxnet as mx
from collections import OrderedDict
import util_layers as utl
import net_util as net_util

WORK_SPACE=utl.WORK_SPACE


def segment_block(inputs,
                  data_params,
                  in_names=None,
                  name='segment',
                  ext='',
                  ext_inputs=None,
                  iter_num=None,
                  arg_params=None,
                  is_train=True,
                  label=None,
                  suppress=1):
    """ input list: current image, flow, last warped label,
    Inputs:
        inputs: ['image'], inputs['label_db'](optinal)
                label_db is label renderede from database
        data_params: parameters for building segmentation
        suppress: the portion of feature channel for training
    """
    if in_names is None:
        in_names = ['data', 'label_db']

    image = inputs[in_names[0]]
    iter_name = '' if iter_num is None else '_' + str(iter_num)
    iter_name = iter_name + ext
    conv_1d_layer = utl.Conv1dLayer(ext=iter_name, params=arg_params)
    conv_bn_layer = utl.ConvBNLayer(ext=iter_name, params=arg_params)
    deconv_bn_layer = utl.DeconvBNLayer(ext=iter_name, params=arg_params)
    s = suppress

    if iter_num is None or iter_num == 0:
        if 'label_db' in in_names:
            # map label to score
            score_bkg = net_util.label_one_hot(inputs['label_db'],
                                               data_params['class_num'])
            score_bkg = mx.symbol.Pooling(data=score_bkg,
                    pool_type="max", kernel=(3, 3), stride=(2, 2),
                    pad=(1,1), name="pool_bkg")
            score_bkg = score_bkg * 10 - 5
        else:
            score_size = [data_params['batch_size'], data_params['class_num']] \
                    + [ sz / 2 for sz in data_params['in_size']]
            score_bkg = mx.symbol.zeros(score_size)

    else:
        score_bkg = ext_inputs[name + '_score' + '_' + str(iter_num - 1) + ext]
        score_bkg = mx.symbol.UpSampling(score_bkg, scale=2,
                num_filter=data_params['class_num'],
                sample_type='nearest', workspace=WORK_SPACE, num_args=1)

    softmax_bkg = mx.symbol.SoftmaxActivation(data=score_bkg, mode='channel')
    conv_bkg = conv_bn_layer(softmax_bkg, 32/s, 1, act=None, name=name + '_bkg_conv1')

    block1 = conv_1d_layer(image, 32/s, 9, name=name + '_block1')
    block1 = mx.symbol.concat(conv_bkg, block1, dim=1)

    block2 = conv_1d_layer(block1, 64/s, 7, name=name + '_block2')
    block2_1 = conv_1d_layer(block2, 64/s, 3, 1, name=name + '_block2_1')

    block3 = conv_1d_layer(block2_1, 128/s, 5, name=name + '_block3')
    block3_1 = conv_1d_layer(block3, 128/s, 3, 1, name=name + '_block3_1')

    block4 = conv_1d_layer(block3_1, 256/s, 5, name=name + '_block4')
    block4_1 = conv_1d_layer(block4, 256/s, 3, 1, name=name + '_block4_1')

    block5 = conv_1d_layer(block4_1, 512/s, 5, name=name + '_block5')
    block5_1 = conv_1d_layer(block5, 512/s, 3, 1, name=name + '_block5_1')

    code = conv_bn_layer(block5_1, 1024/s, name=name + '_snet_conv1')
    score_low = conv_bn_layer(code, data_params['class_num'], 3, 1,
                              act=None,
                              name=name + '_snet_conv2')

    up_score = deconv_bn_layer(score_low,
                    data_params['class_num'], 4, 2,
                    act=None,
                    name=name + '_up_score')

    up_block4 = deconv_bn_layer(block5_1, 256/s, 4, name=name + '_up_block4')
    up_block4 = mx.symbol.concat(up_block4, block4_1, up_score, dim=1)
    up_block3 = deconv_bn_layer(up_block4, 128/s, 4, name=name + '_up_block3')
    up_block3 = mx.symbol.concat(up_block3, block3_1, dim=1)
    up_block2 = deconv_bn_layer(up_block3, 64/s, 4, name=name + '_up_block2')
    up_block2 = mx.symbol.concat(up_block2, block2_1, dim=1)

    code_up = conv_bn_layer(up_block2, 128/s, name=name + '_snet_up_conv1')
    score = conv_bn_layer(code_up, data_params['class_num'], 3, 1,
                          act=None,
                          name=name + '_snet_up_conv2')

    score_low_ls = mx.symbol.UpSampling(score_low, scale=8,
                num_filter=data_params['class_num'],
                sample_type='bilinear', workspace=WORK_SPACE, num_args=1)
    score_bkg = mx.symbol.Pooling(data=score_bkg,
            pool_type="max", kernel=(3,3), stride=(2,2), pad=(1,1))

    score_low_ls = score_low_ls + score_bkg
    score = score + score_bkg

    # add a convlstm here for label ensemble
    score_code = conv_bn_layer(score, 64/s, 3, 1, name=name + '_merge_conv1')
    score = conv_bn_layer(score_code, data_params['class_num'], 3, 1,
                          act=None,
                          name=name + '_merge_conv2')

    if is_train:
        softmax_low = mx.symbol.SoftmaxOutput(
                data=score_low_ls, label=label, multi_output=True,
                use_ignore=True, ignore_label=0, name="softmax_low" + iter_name)
        softmax = mx.symbol.SoftmaxOutput(
                data=score, multi_output=True,
                use_ignore=True, label=label, ignore_label=0, name="softmax" + iter_name)
    else:
        softmax = mx.symbol.SoftmaxActivation(data=score,
                mode='channel')
        softmax_low = mx.symbol.SoftmaxActivation(data=score_low_ls,
                mode='channel')

    Outputs = OrderedDict({name + '_softmax' + iter_name: softmax,
                           name + '_softmax_low' + iter_name: softmax_low,
                           name + '_score' + iter_name: score})

    return Outputs


def refine_block(inputs, data_params, arg_params,
                in_names=None, name='refine',
                scale=4, ext_inputs=None, ext='',
                is_train=False, label=None, suppress=1):
    if in_names is None:
        in_names = ['image', 'segment']

    s = suppress
    image = inputs[in_names[0]]
    conv_bn_layer = utl.ConvBNLayer(ext=ext, params=arg_params)
    deconv_bn_layer = utl.DeconvBNLayer(ext=ext, params=arg_params)

    score_bkg = ext_inputs[in_names[1]]
    score_bkg = mx.symbol.UpSampling(score_bkg, scale=4,
            num_filter=data_params['class_num'],
            sample_type='nearest', workspace=WORK_SPACE, num_args=1)
    softmax_bkg = mx.symbol.SoftmaxActivation(data=score_bkg, mode='channel')
    conv_bkg = conv_bn_layer(softmax_bkg, 32/s, 1, act=None, name=name + '_bkg_conv1')

    conv0 = conv_bn_layer(image, 32/s, name=name + '_conv0')
    conv0 = mx.symbol.concat(conv0, conv_bkg, dim=1)
    conv1 = conv_bn_layer(conv0, 64/s, stride=2, name=name + '_conv1')
    conv1_1 = conv_bn_layer(conv1, 64/s, name=name + '_conv1_1')
    conv2 = conv_bn_layer(conv1_1, 128/s, stride=2, name=name + '_conv2')
    conv2_1 = conv_bn_layer(conv2, 128/s, name=name + '_conv2_1')

    up_conv1 = deconv_bn_layer(conv2_1, 64/s, name=name + '_up_conv1')
    up_conv0 = mx.symbol.concat(up_conv1, conv1_1, dim=1)
    up_conv0 = deconv_bn_layer(up_conv0, 32/s, name=name + '_up_conv0')
    up_conv0 = mx.symbol.concat(up_conv0, conv0, dim=1)

    feat = conv_bn_layer(up_conv0, 32/s, name=name + '_snet_conv1')
    up_score = conv_bn_layer(feat, data_params['class_num'], 1,
            act=None, name=name + '_snet_conv2')

    # up_score = up_score + score_bkg
    if is_train:
        softmax = mx.symbol.SoftmaxOutput(
                data=up_score, label=label, multi_output=True,
                use_ignore=True, ignore_label=0, name="softmax_up")
    else:
        softmax = mx.symbol.SoftmaxActivation(
                data=up_score, mode='channel')
    Outputs = OrderedDict({name + '_softmax': softmax})

    return Outputs


def recurrent_seg_block(inputs,
                        data_params,
                        ext_inputs=None,
                        arg_params=None,
                        name='segment',
                        in_names=['data', 'label_db'],
                        ext='',
                        spatial_steps=2,
                        is_train=True,
                        label=None,
                        is_refine=True,
                        suppress=1):
    if is_train:
        assert label is not None


    # make the shareble parameters
    if arg_params is None:
        block = segment_block(inputs,
                              data_params,
                              in_names=in_names,
                              name=name,
                              ext_inputs=ext_inputs,
                              is_train=False,
                              suppress=suppress)
        arg_params = net_util.def_arguments(
                block[name + '_score'].list_arguments())

    outputs_all = OrderedDict([])
    label_low = net_util.down_sample(label)

    outputs = segment_block(inputs, data_params,
                in_names=in_names,
                name=name,
                ext=ext,
                ext_inputs=ext_inputs,
                iter_num=0,
                arg_params=arg_params,
                is_train=is_train,
                label=label_low,
                suppress=suppress)

    iter_str = '_0'
    loss_all = [outputs[name + '_softmax_low' + iter_str],
                outputs[name + '_softmax' + iter_str]]

    for i in range(1, spatial_steps):
        outputs = segment_block(inputs, data_params, name=name, ext=ext,
               in_names=in_names, ext_inputs=outputs, iter_num=i,
               arg_params=arg_params,is_train=is_train, label=label,
               suppress=suppress)
        iter_str = '_' + str(i)
        loss_all = loss_all +  [outputs[name + '_softmax_low' + iter_str],
                                outputs[name + '_softmax' + iter_str]]

    out_name = name + '_softmax_' + str(spatial_steps - 1)

    # print out_name
    if is_refine:
        target_name = name + '_score' + iter_str
        outputs = refine_block(inputs, data_params,
                in_names=[in_names[0], target_name],
                name=name + '_refine', ext=ext,
                ext_inputs=outputs,
                arg_params=arg_params, is_train=is_train, label=label,
                suppress=suppress)
        loss_all = loss_all + [outputs[name + '_refine_softmax']]
        out_name = name + '_refine_softmax'

    outputs_all[out_name] = outputs[out_name]

    if is_train:
        loss_all = mx.symbol.Group(loss_all)
        outputs_all[name + '_loss'] = loss_all

    return outputs_all



