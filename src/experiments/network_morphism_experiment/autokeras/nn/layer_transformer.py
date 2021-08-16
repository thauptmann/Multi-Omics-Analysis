import numpy as np

from .layers import StubDense, get_n_dim, get_batch_norm_class

from experiments.network_morphism_experiment.autokeras.constant import Constant


def wider_pre_dense(layer, n_add, weighted=True):
    if not weighted:
        return StubDense(layer.input_units, layer.units + n_add)

    n_units2 = layer.units

    teacher_w, teacher_b = layer.get_weights()
    rand = np.random.randint(n_units2, size=n_add)
    student_w = teacher_w.copy()
    student_b = teacher_b.copy()

    # target layer update (i)
    for i in range(n_add):
        teacher_index = rand[i]
        new_weight = teacher_w[teacher_index, :]
        new_weight = new_weight[np.newaxis, :]
        student_w = np.concatenate((student_w, add_noise(new_weight, student_w)), axis=0)
        student_b = np.append(student_b, add_noise(teacher_b[teacher_index], student_b))

    new_pre_layer = StubDense(layer.input_units, n_units2 + n_add)
    new_pre_layer.set_weights((student_w, student_b))

    return new_pre_layer


def wider_bn(layer, start_dim, total_dim, n_add, weighted=True):
    n_dim = get_n_dim(layer)
    if not weighted:
        return get_batch_norm_class(n_dim)(layer.num_features + n_add)

    weights = layer.get_weights()

    new_weights = [add_noise(np.ones(n_add, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.zeros(n_add, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.zeros(n_add, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.ones(n_add, dtype=np.float32), np.array([0, 1]))]

    student_w = tuple()
    for weight, new_weight in zip(weights, new_weights):
        temp_w = weight.copy()
        temp_w = np.concatenate((temp_w[:start_dim], new_weight, temp_w[start_dim:total_dim]))
        student_w += (temp_w,)
    new_layer = get_batch_norm_class(n_dim)(layer.num_features + n_add)
    new_layer.set_weights(student_w)
    return new_layer


def wider_next_dense(layer, start_dim, total_dim, n_add, weighted=True):
    if not weighted:
        return StubDense(layer.input_units + n_add, layer.units)
    teacher_w, teacher_b = layer.get_weights()
    student_w = teacher_w.copy()
    n_units_each_channel = int(teacher_w.shape[1] / total_dim)

    new_weight = np.zeros((teacher_w.shape[0], n_add * n_units_each_channel))
    student_w = np.concatenate((student_w[:, :start_dim * n_units_each_channel],
                                add_noise(new_weight, student_w),
                                student_w[:, start_dim * n_units_each_channel:total_dim * n_units_each_channel]),
                               axis=1)

    new_layer = StubDense(layer.input_units + n_add, layer.units)
    new_layer.set_weights((student_w, teacher_b))
    return new_layer


def add_noise(weights, other_weights):
    w_range = np.ptp(other_weights.flatten())
    noise_range = Constant.NOISE_RATIO * w_range
    noise = np.random.uniform(-noise_range / 2.0, noise_range / 2.0, weights.shape)
    return np.add(noise, weights)


def init_dense_weight(layer):
    units = layer.units
    weight = np.eye(units)
    bias = np.zeros(units)
    layer.set_weights((add_noise(weight, np.array([0, 1])), add_noise(bias, np.array([0, 1]))))


def init_conv_weight(layer):
    n_filters = layer.filters
    filter_shape = (layer.kernel_size, ) * get_n_dim(layer)
    weight = np.zeros((n_filters, n_filters) + filter_shape)

    center = tuple(map(lambda x: int((x - 1) / 2), filter_shape))
    for i in range(n_filters):
        filter_weight = np.zeros((n_filters,) + filter_shape)
        index = (i,) + center
        filter_weight[index] = 1
        weight[i, ...] = filter_weight
    bias = np.zeros(n_filters)

    layer.set_weights((add_noise(weight, np.array([0, 1])), add_noise(bias, np.array([0, 1]))))


def init_bn_weight(layer):
    n_filters = layer.num_features
    new_weights = [add_noise(np.ones(n_filters, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.zeros(n_filters, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.zeros(n_filters, dtype=np.float32), np.array([0, 1])),
                   add_noise(np.ones(n_filters, dtype=np.float32), np.array([0, 1]))]
    layer.set_weights(new_weights)
