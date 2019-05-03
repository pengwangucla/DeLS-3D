# utils for building networks with mxnet

from collections import OrderedDict
import mxnet as mx
import pdb

def def_arguments(arg_list, ignore=['data','label']):
    arg_params = {}
    for args in arg_list:
        for ignore_name in ignore:
            if ignore_name in args:
                continue

        arg_params[args] = mx.sym.Variable(args)
    return arg_params


def load_mxparams_from_file(path):
     """
         Load model checkpoint from file.
     """

     save_dict = mx.nd.load(path)
     arg_params = {}
     aux_params = {}
     for k, v in save_dict.items():
         tp, name = k.split(':', 1)
         if tp == 'arg':
             arg_params[name] = v
         if tp == 'aux':
             aux_params[name] = v

     return arg_params, aux_params


def down_sample(label, scale=4, is_label=False):
    """Nearest down sample for each label
    """
    if scale == 1:
        return label

    if is_label:
        label = mx.sym.expand_dims(label, axis=1)

    label_low = mx.symbol.Pooling(data=label,
                pool_type="avg", kernel=(1,1), stride=(scale,scale),
                pad=(0,0))
    if is_label:
        label_low = mx.sym.sum(label_low, axis=1)

    return label_low


def args_from_models(models, with_symbol=True):
    all_sym = []
    all_args = {}
    all_auxs = {}
    for key in models.keys():
        print 'loading {}'.format(key)
        results = models[key].split('-')
        prefix = '-'.join(results[:-1])
        epoch = results[-1]
        if with_symbol:
            symbol, arg_params, aux_params = mx.model.load_checkpoint(
                    prefix, int(epoch))
            all_sym.append(symbol)
        else:
            model_file = "%s.params" % models[key]
            arg_params, aux_params = load_mxparams_from_file(model_file)
        all_args.update(arg_params)
        all_auxs.update(aux_params)

    return all_sym, all_args, all_auxs


def get_mx_var_by_name(names):
    variables = OrderedDict([])
    for name in names:
        variables[name] = mx.symbol.Variable(name=name)
    return variables


def load_model(models, data_names, data_shapes, net=None, ctx=None):
    """
       load model with fixed length
    """
    with_symbol = net is None
    net_pnt, arg_params, aux_params = args_from_models(models, with_symbol)
    if net is None:
        net = net_pnt[0]

    data_shapes = [(name, shape) for name, shape in zip(data_names, data_shapes)]
    mod = mx.mod.Module(net,
                        data_names=data_names,
                        label_names=None,
                        context=ctx)

    mod.bind(for_training=False, data_shapes=data_shapes)
    if not arg_params:
        mod.init_params()
    else:
        mod.set_params(arg_params, aux_params, allow_missing=False)

    return mod


import numpy as np
def softmax(x, axis):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def euler2mat_v2(rot):
    """Converts euler angles to rotation matrix
       TODO: remove the dimension for 'N' (deprecated for converting all source
             poses altogether)
       Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
      Args:
          Rotation euler angle size = [B, 3]
      Returns:
          Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """

    roll = mx.sym.slice_axis(rot, axis=1, begin=0, end=1)
    pitch = mx.sym.slice_axis(rot, axis=1, begin=1, end=2)
    yaw = mx.sym.slice_axis(rot, axis=1, begin=2, end=3)

    # Expand to B x 1 x 1
    roll = mx.sym.expand_dims(roll, axis=2)
    pitch = mx.sym.expand_dims(pitch, axis=2)
    yaw = mx.sym.expand_dims(yaw, axis=2)


    cosz = mx.sym.cos(yaw)
    sinz = mx.sym.sin(yaw)
    cosy = mx.sym.cos(pitch)
    siny = mx.sym.sin(pitch)
    cosx = mx.sym.cos(roll)
    sinx = mx.sym.sin(roll)

    mat_11 = cosy * cosz
    mat_12 = sinx * siny * cosz - cosx * sinz
    mat_13 = cosx * siny * cosz + sinx * sinz
    mat_1 = mx.sym.concat(mat_11, mat_12, mat_13, dim=2)

    mat_21 = cosy * sinz
    mat_22 = sinx * siny * sinz + cosx * cosz
    mat_23 = cosx * siny * sinz - sinx * cosz
    mat_2 = mx.sym.concat(mat_21, mat_22, mat_23, dim=2)

    mat_31 = -1 * siny
    mat_32 = sinx * cosy
    mat_33 = cosx * cosy
    mat_3 = mx.sym.concat(mat_31, mat_32, mat_33, dim=2)

    mat = mx.sym.concat(mat_1, mat_2, mat_3, dim=1)

    return mat

def euler2mat_grad(rot, grad_last):
    """ nd array derivative from rotation matrix
      Args:
          Rotation euler angle size = [B, 3]
      Returns:
          Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    roll = mx.nd.slice_axis(rot, axis=1, begin=0, end=1)
    pitch = mx.nd.slice_axis(rot, axis=1, begin=1, end=2)
    yaw = mx.nd.slice_axis(rot, axis=1, begin=2, end=3)

    roll = mx.nd.expand_dims(roll, axis=2)
    pitch = mx.nd.expand_dims(pitch, axis=2)
    yaw = mx.nd.expand_dims(yaw, axis=2)

    cosz = mx.nd.cos(yaw)
    sinz = mx.nd.sin(yaw)
    cosy = mx.nd.cos(pitch)
    siny = mx.nd.sin(pitch)
    cosx = mx.nd.cos(roll)
    sinx = mx.nd.sin(roll)

    # for x
    zeros = mx.nd.zeros_like(roll)

    mat_11 = zeros
    mat_12 = cosx * siny * cosz + sinx * sinz
    mat_13 = -1* sinx * siny * cosz + cosx * sinz
    mat_1 = mx.nd.concat(mat_11, mat_12, mat_13, dim=2)

    mat_21 = zeros
    mat_22 = cosx * siny * sinz - sinx * cosz
    mat_23 = -1 * sinx * siny * sinz - cosx * cosz
    mat_2 = mx.nd.concat(mat_21, mat_22, mat_23, dim=2)

    mat_31 = zeros
    mat_32 = cosx * cosy
    mat_33 = -1 * sinx * cosy
    mat_3 = mx.nd.concat(mat_31, mat_32, mat_33, dim=2)

    mat_x = mx.nd.concat(mat_1, mat_2, mat_3, dim=1)

    return mx.nd.sum(mat_x, axis=[1, 2])


def euler2mat(rot):
    """Converts euler angles to rotation matrix
       Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
      Args:
          Rotation euler angle size = [B, 3]
      Returns:
          Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """

    roll = mx.sym.slice_axis(rot, axis=1, begin=0, end=1)
    pitch = mx.sym.slice_axis(rot, axis=1, begin=1, end=2)
    yaw = mx.sym.slice_axis(rot, axis=1, begin=2, end=3)

    # Expand to B x 1 x 1
    roll = mx.sym.expand_dims(roll, axis=2)
    pitch = mx.sym.expand_dims(pitch, axis=2)
    yaw = mx.sym.expand_dims(yaw, axis=2)

    zeros = mx.sym.zeros_like(roll)
    ones  = mx.sym.ones_like(roll)

    cosz = mx.sym.cos(yaw)
    sinz = mx.sym.sin(yaw)
    yaw_mat_1 = mx.sym.concat(cosz, -1 * sinz, zeros, dim=2)
    yaw_mat_2 = mx.sym.concat(sinz,  cosz, zeros, dim=2)
    yaw_mat_3 = mx.sym.concat(zeros, zeros, ones, dim=2)
    yaw_mat = mx.sym.concat(yaw_mat_1, yaw_mat_2, yaw_mat_3, dim=1)

    cosy = mx.sym.cos(pitch)
    siny = mx.sym.sin(pitch)
    pitch_mat_1 = mx.sym.concat(cosy, zeros, siny, dim=2)
    pitch_mat_2 = mx.sym.concat(zeros, ones, zeros, dim=2)
    pitch_mat_3 = mx.sym.concat(-1 * siny, zeros, cosy, dim=2)
    pitch_mat = mx.sym.concat(pitch_mat_1, pitch_mat_2, pitch_mat_3, dim=1)

    # Expand to B x 3 x 3
    cosx = mx.sym.cos(roll)
    sinx = mx.sym.sin(roll)
    roll_mat_1 = mx.sym.concat(ones, zeros, zeros, dim=2)
    roll_mat_2 = mx.sym.concat(zeros, cosx, -1 * sinx, dim=2)
    roll_mat_3 = mx.sym.concat(zeros, sinx, cosx, dim=2)
    roll_mat = mx.sym.concat(roll_mat_1, roll_mat_2, roll_mat_3, dim=1)

    rotMat = mx.sym.batch_dot(yaw_mat, pitch_mat)
    rotMat = mx.sym.batch_dot(rotMat, roll_mat)

    return rotMat


def pose2mat(pose):
    """Converts 6DoF pose to transformation matrix
      Args:
          vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
      Returns:
          A transformation matrix -- [B, 3, 4]
    """
    trans = mx.sym.slice_axis(pose, axis=1, begin=0, end=3)
    trans = mx.symbol.expand_dims(trans, axis=2)
    rot = mx.sym.slice_axis(pose, axis=1, begin=3, end=6)
    rot_mat = euler2mat_v2(rot)
    mat = mx.sym.concat(rot_mat, trans, dim=2)
    return mat


def world2cam(points, proj):
    """Transforms coordinates in a camera frame to the pixel frame.
      Args:
        cam_coords: [batch, 4, point_num]
        proj: [batch, 3, 4]
      Returns:
        Pixel coordinates projected from the camera frame [batch, 2, height, width]
    """
    cam_coor = mx.symbol.batch_dot(proj, points)
    # x_coor = mx.sym.slice_axis(cam_coor, axis=1, begin=0, end=1)
    # y_coor = mx.sym.slice_axis(cam_coor, axis=1, begin=1, end=2)
    # depth = mx.sym.slice_axis(cam_coor, axis=1, begin=2, end=3)

    # # clip very small depth
    # depth = mx.sym.maximum(depth, 0.5)

    # x_coor = x_coor / depth
    # y_coor = y_coor / depth
    # pixel_coor = mx.symbol.concat(x_coor, y_coor, dim=1)

    # return pixel_coor
    return cam_coor


def test_grad_euler2mat(pose_in, var_id, delta):
    pose = mx.sym.ones([1, 6])
    points = mx.sym.ones([1, 4, 3, 3])
    points = mx.sym.reshape(points, shape=[1, 4, 9])

    pose = mx.sym.Variable('pose')
    mat = pose2mat(pose)
    loss = mx.sym.MakeLoss(data = mx.sym.sum(mat, axis=[1,2]),
            grad_scale=1.0)

    pose_np = mx.nd.array(pose_in)
    executor = loss.simple_bind(ctx=mx.cpu(), pose=(1,6))
    executor.forward(is_train=True, pose=pose_np)

    print executor.outputs
    out_0 = executor.outputs[0].asnumpy()

    executor.backward()
    print executor.grad_arrays
    grad_all = executor.grad_arrays[0].asnumpy()[0]

    # use delta for gradient check
    delta_all = np.zeros((1, 6), dtype=np.float32)
    delta_all[:, var_id] = delta
    perturb = np.array(delta_all)
    pose_np_add = mx.nd.array(pose_ar + perturb)
    ex = loss.simple_bind(ctx=mx.cpu(), pose=(1,6))
    ex.forward(is_train=True, pose=pose_np_add)
    out_1 = ex.outputs[0].asnumpy()
    grad = (out_1 - out_0) / delta
    print out_1, out_0, grad
    print grad - grad_all[var_id]

    # use formula for gradient check
    grad_last = mx.nd.array(np.ones((1, 3, 3)))
    grad_form = euler2mat_grad(pose_np[:, 3:], grad_last)
    print "grad form", grad_form


def test_grad_world2cam(mat_in, var_id, delta):

    points = mx.sym.Variable('points')
    mat = mx.sym.Variable('mat')

    points_np = mx.nd.array(np.ones([1, 4, 3 * 3]))
    mat_np = mx.nd.array(mat_in)

    points = mx.sym.stop_gradient(points)
    loc = world2cam(points, mat)
    loss = mx.sym.MakeLoss(data=mx.sym.sum(loc, axis=[1,2]),
            grad_scale=1.0)
    ex = loss.simple_bind(ctx=mx.cpu(), mat=(1, 3, 4), points=(1, 4, 3*3))
    ex.forward(is_train=True, mat=mat_np, points=points_np)
    print ex.outputs

    ex.backward()
    print ex.grad_arrays
    grad_all = ex.grad_arrays[0].asnumpy()[0]

    # use delta for gradient check
    delta_all = np.zeros((1, 3, 4), dtype=np.float32)
    delta_all[:, var_id[0], var_id[1]] = delta
    perturb = mx.nd.array(delta_all)
    mat_nd_add = mat_np + perturb
    mat_nd_minus = mat_np - perturb

    ex.forward(is_train=True, mat=mat_nd_add, points=points_np)
    out_add = ex.outputs[0].asnumpy()

    ex.forward(is_train=True, mat=mat_nd_minus, points=points_np)
    out_minus = ex.outputs[0].asnumpy()
    grad = (out_add - out_minus) / (2 * delta)
    print out_add, out_minus, grad
    print grad - grad_all[var_id[0], var_id[1]]


def label_one_hot(label, num_classes):

    label_smooth = mx.sym.sum(label, axis=1)
    label_smooth = mx.sym.one_hot(label_smooth, depth=num_classes)
    label_smooth = mx.sym.transpose(label_smooth, (0, 3, 1, 2))
    label_smooth = mx.sym.stop_gradient(label_smooth)

    return label_smooth



if __name__ == '__main__':
    pose_ar = np.ones([1, 6])
    var_id = 3
    delta = 1e-4

    # test_grad_euler2mat(pose_ar, var_id, delta)
    mat = np.ones([1, 3, 4])
    var_id = [0, 0]
    delta = 1e-4
    test_grad_world2cam(mat, var_id, delta)



