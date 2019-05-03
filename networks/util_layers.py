import mxnet as mx
from collections import OrderedDict

WORK_SPACE=1024

def fcnxs_score(input,
                crop,
                offset,
                kernel=(64,64),
                stride=(32,32),
                numclass=21,
                name='score',
                workspace_default=WORK_SPACE):
    # score out
    bigscore = mx.symbol.Deconvolution(data=input,
            kernel=kernel,
            stride=stride,
            adj=(stride[0]-1, stride[1]-1),
            num_filter=numclass,
            workspace=workspace_default,
            name=name)

    upscore = mx.symbol.Crop(*[bigscore, crop], offset=offset, name="upscore")
    softmax = mx.symbol.SoftmaxOutput(data=upscore,
            multi_output=True, use_ignore=True, ignore_label=255, name="softmax")
    return softmax


def get_value(pm, name):
    if pm and (name in pm):
        return pm[name]
    return None


class Conv1dLayer(object):
    # ext is use for different layer name while sharing the weight
    def __init__(self, ext='', params=None):
        self.ext=ext
        self.params=params

    def __call__(self,
                 input,
                 num_filter,
                 filter_size=3,
                 stride=2,
                 act='LeakyRELU',
                 name='conv_1d'):

        pm = self.params
        ext = self.ext

        y_name=name + '_y'
        conv = mx.symbol.Convolution(data=input,
                                    weight=get_value(pm, y_name + '_weight'),
                                    bias=get_value(pm, y_name + '_bias'),
                                    num_filter=num_filter,
                                    kernel=(filter_size, 1),
                                    stride=(stride, 1),
                                    pad=((filter_size-1)/2, 0),
                                    name=y_name + ext)
        if act == 'LeakyRELU':
            out = mx.symbol.LeakyReLU(data=conv,
                                      act_type='rrelu',
                                      name='rrelu_%s' % (name + '_y' + ext))
        else:
            out = conv

        x_name = name + '_x'
        conv = mx.symbol.Convolution(data=out,
                                    weight=get_value(pm, x_name + '_weight'),
                                    bias=get_value(pm, x_name + '_bias'),
                                    num_filter=num_filter,
                                    kernel=(1, filter_size),
                                    stride=(1, stride),
                                    pad=(0, (filter_size-1)/2),
                                    name=name + '_x' + ext)
        if act == 'LeakyRELU':
            out = mx.symbol.LeakyReLU(data=conv,
                                      act_type='rrelu',
                                      name='rrelu_%s' % (name + '_x' + ext))
        else:
            out = conv

        return out


class ConvBNLayer(object):
    def __init__(self,  ext='', params=None):
        self.ext=ext
        self.params=params

    def __call__(self,
                 input,
                 num_filter,
                 filter_size=3,
                 stride=1,
                 act='LeakyRELU',
                 name='conv_bn'):
        pm = self.params
        ext=self.ext

        conv = mx.symbol.Convolution(data=input,
                                     weight=get_value(pm, name + '_weight'),
                                     bias=get_value(pm, name + '_bias'),
                                     num_filter=num_filter,
                                     kernel=(filter_size, filter_size),
                                     stride=(stride, stride),
                                     pad=((filter_size-1)/2, (filter_size-1)/2),
                                     workspace=1024,
                                     name=name+ext)
        if act is None:
            out = conv
        elif act == 'LeakyRELU':
            out = mx.symbol.LeakyReLU(data=conv,
                                      act_type='rrelu',
                                      name='rrelu_%s' % (name + ext))
        elif act == 'RELU':
            out = mx.symbol.relu(data=conv)

        return out


class DeconvBNLayer(object):
    def __init__(self, ext='', params=None, no_bias=True):
        self.ext=ext
        self.params=params
        self.no_bias=no_bias

    def __call__(self,
                 input,
                 num_filter,
                 filter_size=4,
                 stride=2,
                 act='LeakyRELU',
                 name='deconv_bn'):
        pm=self.params
        ext=self.ext
        bias = None
        if self.no_bias == False:
            bias = get_value(pm, name + '_bias')

        conv = mx.symbol.Deconvolution(data=input,
                                       weight=get_value(pm, name + '_weight'),
                                       bias=bias,
                                       kernel=(filter_size, filter_size),
                                       stride=(stride, stride),
                                       num_filter=num_filter,
                                       pad=((filter_size - 1)/2, (filter_size - 1)/2),
                                       no_bias=self.no_bias,
                                       workspace=WORK_SPACE,
                                       name=name + ext)  # 2X
        if act is None:
            out = conv
        elif act == 'LeakyRELU':
            out = mx.symbol.LeakyReLU(data=conv,
                                      act_type='rrelu',
                                      name='rrelu_%s' % (name + ext))
        elif act == 'RELU':
            out = mx.symbol.relu(data=conv, name='relu_%s' % (name + ext))

        return out


class FCBNLayer(object):
    def __init__(self, ext='', params=None):
        self.ext=ext
        self.params=params

    def __call__(self,
                 input,
                 num_hidden,
                 act='LeakyRELU',
                 name='fc_bn'):
        pm=self.params
        ext=self.ext

        conv = mx.symbol.FullyConnected(data=input,
                                       weight=pm[name + '_weight'] if pm else None,
                                       bias=pm[name + '_bias'] if pm else None,
                                       num_hidden=num_hidden,
                                       name=name + ext)  # 2X
        if act is None:
            out = conv
        elif act == 'LeakyRELU':
            out = mx.symbol.LeakyReLU(data=conv,
                                      act_type='rrelu',
                                      name='rrelu_%s' % (name + ext))
        elif act == 'RELU':
            out = mx.symbol.relu(data=conv, name='relu_%s' % (name + ext))

        return out



def conv_block(image, arg_params=None, name='flow', ext=''):
    """a convolutional block for extracting feature
    """

    conv_1d_layer = Conv1dLayer(ext=ext, params=arg_params)
    block1 = conv_1d_layer(image, 32, 9, name=name + '_block1')
    block2 = conv_1d_layer(block1, 32, 7, name=name + '_block2')
    block2_1 = conv_1d_layer(block2, 32, 3, 1, name=name + '_block2_1')

    block3 = conv_1d_layer(block2_1, 64, 5, name=name + '_block3')
    block3_1 = conv_1d_layer(block3, 64, 3, 1, name=name + '_block3_1')

    block4 = conv_1d_layer(block3_1, 128, 5, name=name + '_block4')
    block4_1 = conv_1d_layer(block4, 128, 3, 1, name=name + '_block4_1')

    block5 = conv_1d_layer(block4_1, 256, 5, name=name + '_block5')
    block5_1 = conv_1d_layer(block5, 256, 3, 1, name=name + '_block5_1')

    out = [('block5_1', block5_1)]
    for i in range(1, 6):
      exec('out.append((\'block' + str(i) +'\', block' + str(i) + '))')
      if i > 1:
        exec('out.append((\'block' + str(i) +'_1\', block' + str(i) + '_1))')

    return OrderedDict(out)
