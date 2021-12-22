import torch
from ax import Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep


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