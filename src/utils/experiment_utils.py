import torch
from ax import Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep

from utils.network_training_util import calculate_mean_and_std_auc


def create_generation_strategy(sampling_method, sobol_iterations, random_seed):
    if sampling_method == 'gp':
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL,
                               num_trials=sobol_iterations,
                               max_parallelism=1,
                               model_kwargs={"seed": random_seed}),
                GenerationStep(
                    model=Models.BOTORCH,
                    max_parallelism=1,
                    num_trials=-1,
                ),
            ],
            name="Sobol+GPEI"
        )
    elif sampling_method == 'saasbo':
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL,
                               num_trials=sobol_iterations,
                               max_parallelism=1,
                               model_kwargs={"seed": random_seed}),
                GenerationStep(
                    model=Models.FULLYBAYESIAN,
                    num_trials=-1,
                    max_parallelism=1,
                    model_kwargs={
                        "num_samples": 256,
                        "warmup_steps": 512,
                        "disable_progbar": True,
                        "torch_device": torch.device('cpu'),
                        "torch_dtype": torch.double,
                    },
                ),
            ],
            name="SAASBO"
        )
    else:
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=-1),
            ],
            name="Sobol"
        )
    return generation_strategy


def write_results_to_file(drug_name, extern_auc_list, extern_auprc_list, result_file, test_auc_list, test_auprc_list):
    result_dict = {
        'test auroc': test_auc_list,
        'test auprc': test_auprc_list,
        'extern auroc': extern_auc_list,
        'extern auprc': extern_auprc_list
    }
    result_file.write(f'\n test auroc list: {test_auc_list} \n')
    result_file.write(f'\n test auprc list: {test_auprc_list} \n')
    result_file.write(f'\n extern auroc list: {extern_auc_list} \n')
    result_file.write(f'\n extern auprc list: {extern_auprc_list} \n')
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)