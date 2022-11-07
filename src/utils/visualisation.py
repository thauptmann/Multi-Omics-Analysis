import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("whitegrid")


def save_auroc_plots(all_aucs, path, iteration, model_transitions=None):
    best_aucs = np.maximum.accumulate(all_aucs, axis=0)

    x = list(range(1, len(all_aucs) + 1))
    file_names = ("all", "best")
    for y, name in zip((all_aucs, best_aucs), file_names):
        ax = sns.lineplot(x=x, y=y)
        ax.set(
            xlabel="Iteration",
            ylabel="Auroc",
            title="Model performance vs. # of iterations",
        )
        if model_transitions is not None and model_transitions > 0:
            ax.vlines(
                x=model_transitions, ymin=min(y), ymax=max(y), linestyles="dashed"
            )
        ax.get_figure().savefig(
            str(path / f"{name}_multi-omics_iteration_{iteration}.svg")
        )
        ax.get_figure().clf()


def save_auroc_with_variance_plots(aucs_list, path, iteration, model_transitions=None):
    mean_aucs = np.mean(aucs_list, axis=0)
    std_aucs = np.std(aucs_list, axis=0)
    y_upper = [1 if i > 1 else i for i in mean_aucs + std_aucs]
    y_lower = mean_aucs - std_aucs
    best_aucs = np.maximum.accumulate(mean_aucs)
    x = list(range(1, len(mean_aucs) + 1))

    # best
    ax = sns.lineplot(x=x, y=best_aucs)
    ax.set(
        xlabel="Iteration",
        ylabel="Auroc",
        title="Model performance vs. # of iterations",
    )
    if model_transitions is not None and model_transitions > 0:
        ax.vlines(
            x=model_transitions,
            ymin=min(best_aucs),
            ymax=max(best_aucs),
            linestyles="dashed",
        )
    ax.get_figure().savefig(str(path / f"best_multi-omics_iteration_{iteration}.svg"))
    ax.get_figure().clf()

    # all
    upper_and_lower = np.concatenate([y_upper, y_lower])
    ax = sns.lineplot(x=x, y=mean_aucs)
    ax.set(
        xlabel="Iteration",
        ylabel="Auroc",
        title="Model performance vs. # of iterations",
    )
    ax.fill_between(x, y_lower, y_upper, alpha=0.2)

    if model_transitions is not None and model_transitions > 0:
        ax.vlines(
            x=model_transitions,
            ymin=min(upper_and_lower),
            ymax=max(upper_and_lower),
            linestyles="dashed",
        )
    ax.get_figure().savefig(str(path / f"all_multi-omics_iteration_{iteration}.svg"))
    ax.get_figure().clf()


# Helper method to print importances and visualize distribution
def visualize_importances(
    feature_names,
    importances,
    targets,
    predictions,
    feature_values,
    title="Average Feature Importances",
    axis_title="Features",
    number_of_most_important_features=10,
    path="",
    file_name="",
):
    number_of_most_important_features += 1
    mean_importances = np.mean(importances, axis=0)
    sd_importances = np.std(importances, axis=0)
    sorted_indices = mean_importances.argsort()
    highest_indices = sorted_indices[-1:-number_of_most_important_features:-1].copy()
    most_important_features = feature_names[highest_indices]
    highest_importances = mean_importances[highest_indices]
    highest_importance_sd = sd_importances[highest_indices]

    absolute_sorted_indices = (np.abs(mean_importances)).argsort()
    absolute_highest_indices = absolute_sorted_indices[
        -1:-number_of_most_important_features:-1
    ].copy()
    absolute_most_important_features = feature_names[absolute_highest_indices]
    absolute_highest_importances = mean_importances[absolute_highest_indices]
    absolute_highest_importance_sd = sd_importances[absolute_highest_indices]

    negative_mean_importances = np.mean(importances * -1, axis=0)
    negative_sd_importances = np.std(importances, axis=0)
    negative_sorted_indices = negative_mean_importances.argsort()
    negative_highest_indices = negative_sorted_indices[
        -1:-number_of_most_important_features:-1
    ].copy()
    negative_most_important_features = feature_names[negative_highest_indices]
    negative_highest_importances = negative_mean_importances[negative_highest_indices]
    negative_highest_importance_sd = negative_sd_importances[negative_highest_indices]

    draw_attributions(
        title,
        axis_title,
        path,
        file_name + "_positive",
        most_important_features,
        highest_importances,
        highest_importance_sd,
    )

    draw_swarm_attributions(
        path,
        file_name,
        feature_names[absolute_highest_indices],
        importances[:10, absolute_highest_indices],
        feature_values[:10, absolute_highest_indices],
    )

    sum_of_rest = np.sum(
        mean_importances[sorted_indices[:-number_of_most_important_features]]
    )
    most_important_features = np.append(most_important_features, "Remaining features")
    highest_importances = np.append(highest_importances, sum_of_rest)
    highest_importance_sd = np.append(highest_importance_sd, 0)

    draw_attributions(
        title,
        axis_title,
        path,
        file_name + "_with_rest",
        most_important_features,
        highest_importances,
        highest_importance_sd,
    )

    draw_attributions(
        title,
        axis_title,
        path,
        file_name + "_absolute",
        absolute_most_important_features,
        absolute_highest_importances,
        absolute_highest_importance_sd,
    )

    draw_attributions(
        title,
        axis_title,
        path,
        file_name + "_negative",
        negative_most_important_features,
        negative_highest_importances,
        negative_highest_importance_sd,
    )


def draw_attributions(
    title,
    axis_title,
    path,
    file_name,
    most_important_features,
    highest_importances,
    highest_importance_sd,
):
    ax = sns.barplot(x=most_important_features, y=highest_importances, color="b")
    ax.set_xlabel(axis_title)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    plt.errorbar(x=x_coords, y=y_coords, yerr=highest_importance_sd, fmt="none", c="k")

    fig = ax.get_figure()
    fig.savefig(str(path / f"{file_name}.pdf"), bbox_inches="tight")
    fig.clf()


def draw_swarm_attributions(
    path,
    file_name,
    most_important_features,
    highest_importances,
    features_values,
):
    # create df to make it easier
    number_of_samples = len(features_values)
    multiplied_important_features = np.tile(most_important_features, number_of_samples)
    d = {
        "Feature Name": multiplied_important_features,
        "Value": features_values.flatten(),
        "Attribution": highest_importances.flatten(),
    }
    df = pd.DataFrame(d)
    ax = sns.stripplot(
        data=df,
        x="Attribution",
        y="Feature Name",
        hue="Value",
        palette="viridis",
        size=5,
    )
    ax.set_xlabel("Attribution")

    # norm = plt.Normalize(tips['size'].min(), tips['size'].max())

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])   

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    fig = ax.get_figure()
    fig.savefig(str(path / f"{file_name}_swarm.pdf"), bbox_inches="tight")
    fig.clf()
