import numpy as np
import plotly.graph_objects as go


def save_auroc_plots(all_aucs, path, model_transitions=None):

    best_aucs = np.maximum.accumulate(all_aucs, axis=0)
    layout = go.Layout(
        title="Model performance vs. # of iterations",
        title_x=0.5,
        showlegend=True,
        yaxis={"title": 'AUROC',
               "range": (0.5, 1)},
        xaxis={"title": "Iteration"},
    )

    x = list(range(1, len(all_aucs)+1))
    file_names = ('all', 'best')
    for y, name in zip((all_aucs, best_aucs), file_names):
        scatter = go.Scatter(name="mean", mode="lines", x=x, y=y)
        data = [scatter]

        fig = go.Figure(layout=layout, data=data)
        if model_transitions is not None and model_transitions > 0:
            fig.add_vline(x=model_transitions, line_dash='dot')
        fig.write_image(str(path / f'{name}_multi-omics.svg'))

