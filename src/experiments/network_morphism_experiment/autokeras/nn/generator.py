from abc import abstractmethod

from experiments.network_morphism_experiment.autokeras.constant import Constant
from experiments.network_morphism_experiment.autokeras.nn.graph import Graph
from experiments.network_morphism_experiment.autokeras.nn.layers import StubDense, StubReLU, get_dropout_class, \
    get_batch_norm_class, StubConcatenate


class DenseNetGenerator:
    def __init__(self, n_output_node, input_shape_list):
        # DenseNet Constant
        self.num_init_features = 32
        self.dropout_rate = 0.1
        self.input_shape_list = input_shape_list
        self.number_of_outputs = n_output_node

        # Stub layers
        self.n_dim = 1
        self.dropout = get_dropout_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_width):
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        graph = Graph(self.input_shape_list, False)
        # Input layer for every input
        output_nodes = []
        concat_nodes = []

        for i, input_shape in enumerate(self.input_shape_list):
            output_id = self._dense_layer(input_shape, model_width, self.dropout_rate, graph, i)
            concat_nodes.append(model_width)
            output_nodes.append(output_id)
        # classification linear layer
        concat_output_id = graph.add_layer(StubConcatenate(concat_nodes), output_nodes)
        self._dense_layer(sum(concat_nodes), self.number_of_outputs, self.dropout_rate, graph, concat_output_id)
        return graph

    def _dense_layer(self, num_input_features, output_size, dropout_rate, graph, input_node_id):
        out = graph.add_layer(StubDense(num_input_features, output_size), input_node_id)
        out = graph.add_layer(self.batch_norm(output_size), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.dropout(rate=dropout_rate), out)
        return out


