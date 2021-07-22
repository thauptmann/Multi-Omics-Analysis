from abc import abstractmethod

from experiments.network_morphism_experiment.autokeras.constant import Constant
from experiments.network_morphism_experiment.autokeras.nn.graph import Graph
from experiments.network_morphism_experiment.autokeras.nn.layers import StubDense, StubReLU, get_dropout_class, \
 get_batch_norm_class


class DenseNetGenerator:
    def __init__(self, n_output_node, input_shape_list):
        super().__init__(n_output_node, input_shape_list)
        # DenseNet Constant
        self.num_init_features = 32
        self.dropout_rate = 0.1
        self.input_shape_list = input_shape_list
        self.number_of_outputs = n_output_node

        # Stub layers
        self.n_dim = len(self.input_shape_list[0]) - 1
        self.dropout = get_dropout_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_width=None):
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        graph = Graph(self.input_shape_list, False)
        # Input layer for every input
        input_node_ids = range(len(self.input_shape_list))
        for input_node_id in input_node_ids:
            self._dense_layer(self.n_dim, model_width, self.dropout_rate, graph, input_node_id)
        # append elements
        num_features = self.num_init_features
        # classification linear layer
        self._dense_layer(self.n_dim, model_width, self.dropout_rate, graph, input_node_id)
        return graph

    def _dense_layer(self, num_input_features, output_size, dropout_rate, graph, input_node_id):
        out = graph.add_layer(StubDense(num_input_features, output_size), input_node_id)
        out = graph.add_layer(self.batch_norm(output_size), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.dropout(rate=dropout_rate), out)
        return out


