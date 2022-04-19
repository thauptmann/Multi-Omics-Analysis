import numpy as np
import seaborn as sns
sns.set_style('whitegrid')


def save_auroc_plots(all_aucs, path, iteration, model_transitions=None):
    best_aucs = np.maximum.accumulate(all_aucs, axis=0)

    x = list(range(1, len(all_aucs)+1))
    file_names = ('all', 'best')
    for y, name in zip((all_aucs, best_aucs), file_names):
        ax = sns.lineplot(x=x, y=y)
        ax.set(xlabel='Iteration', ylabel='Auroc', title="Model performance vs. # of iterations")
        if model_transitions is not None and model_transitions > 0:
            ax.vlines(x=model_transitions, ymin=min(y), ymax=max(y), linestyles='dashed')
        ax.get_figure().savefig(str(path / f'{name}_multi-omics_iteration_{iteration}.svg'))
        ax.get_figure().clf()


def save_auroc_with_variance_plots(aucs_list, path, iteration, model_transitions=None):
    mean_aucs = np.mean(aucs_list, axis=0)
    std_aucs = np.std(aucs_list, axis=0)
    y_upper = [1 if i > 1 else i for i in mean_aucs + std_aucs]
    y_lower = mean_aucs - std_aucs
    best_aucs = np.maximum.accumulate(mean_aucs)
    x = list(range(1, len(mean_aucs)+1))

    # best
    ax = sns.lineplot(x=x, y=best_aucs)
    ax.set(xlabel='Iteration', ylabel='Auroc', title="Model performance vs. # of iterations")
    if model_transitions is not None and model_transitions > 0:
        ax.vlines(x=model_transitions, ymin=min(best_aucs), ymax=max(best_aucs), linestyles='dashed')
    ax.get_figure().savefig(str(path / f'best_multi-omics_iteration_{iteration}.svg'))
    ax.get_figure().clf()

    # all
    upper_and_lower = np.concatenate([y_upper, y_lower])
    ax = sns.lineplot(x=x, y=mean_aucs)
    ax.set(xlabel='Iteration', ylabel='Auroc', title="Model performance vs. # of iterations")
    ax.fill_between(x, y_lower, y_upper, alpha=0.2)

    if model_transitions is not None and model_transitions > 0:
        ax.vlines(x=model_transitions, ymin=min(upper_and_lower), ymax=max(upper_and_lower), linestyles='dashed')
    ax.get_figure().savefig(str(path / f'all_multi-omics_iteration_{iteration}.svg'))
    ax.get_figure().clf()
