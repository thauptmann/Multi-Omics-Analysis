import argparse


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_iterations', default=200, type=int)
    parser.add_argument('--sobol_iterations', default=50, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--combination', default=None, type=int)
    parser.add_argument('--sampling_method', default='sobol', choices=['gp', 'sobol', 'saasbo'])
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--small_search_space', default=False, action='store_true')
    parser.add_argument('--deactivate_skip_bad_iterations', default=False, action='store_true')
    parser.add_argument('--drug', default='all', choices=['Gemcitabine_tcga', 'Gemcitabine_pdx', 'Cisplatin',
                                                          'Docetaxel', 'Erlotinib', 'Cetuximab', 'Paclitaxel'])
    parser.add_argument('--semi_hard_triplet', default=False, action='store_true')
    parser.add_argument('--deactivate_elbow_method', default=False, action='store_true')
    parser.add_argument('--deactivate_triplet_loss', default=False, action='store_true')

    parser.add_argument('--noisy', default=False, action='store_true')
    parser.add_argument('--use_reconstruction_loss', default=False, action='store_true')
    parser.add_argument('--stack_sigmoid', default=False, action='store_true')

    parser.add_argument('--architecture', default=None, choices=['supervised-ae'])
    return parser.parse_args()
