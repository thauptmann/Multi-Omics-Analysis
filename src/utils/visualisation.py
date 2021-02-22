from ax.plot.trace import optimization_trace_single_method
import numpy as np
import plotly.graph_objects as go


def save_auroc_plots(all_aucs, path, method, model_transitions=None):
    all_auc_plot = optimization_trace_single_method(
        y=all_aucs,
        title="Model performance vs. # of iterations",
        ylabel="AUROC",
        model_transitions=model_transitions
    )

    best_auc_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(all_aucs, axis=1),
        title="Best model performance vs. # of iterations",
        ylabel="AUROC",
        model_transitions=model_transitions
    )

    file_names = ('all', 'best')
    plots = all_auc_plot, best_auc_plot
    for plot, name in zip(plots, file_names):
        data = plot[0]['data']
        lay = plot[0]['layout']
        fig = {
            "data": data,
            "layout": lay,
        }
        fig = go.Figure(fig)
        fig.update_layout(yaxis_range=[0.5, 1])
        fig.write_html(str(path / f'{method}_{name}_multi-omics.html'))
        fig.write_image(str(path / f'{method}_{name}_multi-omics.svg'))
