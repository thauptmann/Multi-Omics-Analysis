import itertools
import os
import pickle
import random
import string
import tempfile

from experiments.network_morphism_experiment.autokeras.constant import Constant


class NoImprovementError(Exception):
    def __init__(self, message):
        self.message = message


def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_file_dir(path):
    """Create path if it does not exist."""
    ensure_dir(os.path.dirname(path))


def has_file(path):
    """Check if the given path exists."""
    return os.path.exists(path)


def pickle_from_file(path):
    """Load the pickle file from the provided path and returns the object."""
    return pickle.load(open(path, 'rb'))


def pickle_to_file(obj, path):
    """Save the pickle file to the specified path."""
    pickle.dump(obj, open(path, 'wb'))


def temp_path_generator():
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokeras')
    return path


def rand_temp_folder_generator():
    """Create and return a temporary directory with the path name '/temp_dir_name/autokeras' (E:g:- /tmp/autokeras)."""
    chars = string.ascii_uppercase + string.digits
    size = 6
    random_suffix = ''.join(random.choice(chars) for _ in range(size))
    sys_temp = temp_path_generator()
    path = sys_temp + '_' + random_suffix
    ensure_dir(path)
    return path


def assert_search_space(search_space):
    grid = search_space
    value_list = []
    if Constant.LENGTH_DIM not in list(grid.keys()):
        print('No length dimension found in search Space. Using default values')
        grid[Constant.LENGTH_DIM] = Constant.DEFAULT_LENGTH_SEARCH
    elif not isinstance(grid[Constant.LENGTH_DIM][0], int):
        print('Converting String to integers. Next time please make sure to enter integer values for Length Dimension')
        grid[Constant.LENGTH_DIM] = list(map(int, grid[Constant.LENGTH_DIM]))

    if Constant.WIDTH_DIM not in list(grid.keys()):
        print('No width dimension found in search Space. Using default values')
        grid[Constant.WIDTH_DIM] = Constant.DEFAULT_WIDTH_SEARCH
    elif not isinstance(grid[Constant.WIDTH_DIM][0], int):
        print('Converting String to integers. Next time please make sure to enter integer values for Width Dimension')
        grid[Constant.WIDTH_DIM] = list(map(int, grid[Constant.WIDTH_DIM]))

    grid_key_list = list(grid.keys())
    grid_key_list.sort()
    for key in grid_key_list:
        value_list.append(grid[key])

    dimension = list(itertools.product(*value_list))
    # print(dimension)
    return grid, dimension


def verbose_print(new_father_id, new_graph, new_model_id):
    """Print information about the operation performed on father model to obtain current model and father's id."""
    cell_size = [24, 49]
    print('New Model Id - ' + str(new_model_id))
    header = ['Father Model ID', 'Added Operation']
    line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(header))
    print('\n' + '+' + '-' * len(line) + '+')
    print('|' + line + '|')
    print('+' + '-' * len(line) + '+')
    for i in range(len(new_graph.operation_history)):
        if i == len(new_graph.operation_history) // 2:
            r = [str(new_father_id), ' '.join(str(item) for item in new_graph.operation_history[i])]
        else:
            r = [' ', ' '.join(str(item) for item in new_graph.operation_history[i])]
        line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(r))
        print('|' + line + '|')
    print('+' + '-' * len(line) + '+')


def validate_xy(x_train, y_train):
    """Validate `x_train`'s type and the shape of `x_train`, `y_train`."""
    try:
        x_train = x_train.astype('float64')
    except ValueError:
        raise ValueError('x_train should only contain numerical data.')

    if len(x_train.shape) < 2:
        raise ValueError('x_train should at least has 2 dimensions.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('x_train and y_train should have the same number of instances.')
