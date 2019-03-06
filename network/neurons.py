import numpy as np


class BaseNeuron:
    def __init__(self, num_weights=None, bias=None):
        self.weights, self.bias = self.generate_weights_and_bias(num_weights, bias)

    @staticmethod
    def generate_weights_and_bias(num_weights, bias):
        weights = np.array(
            [
                np.random.normal()
                for _ in range(num_weights if num_weights is not None else 0)
            ]
        )
        bias = bias if bias is not None else np.random.normal()
        return weights, bias

    def update_weights(self, *args):
        self.weights = np.array([w for w in args])

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def activation(self, value):
        return self.sigmoid(value)

    def deriv_activation(self, value):
        fx = self.activation(value)
        return fx * (1 - fx)

    def sum_feed(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def feedforward(self, inputs):
        total = self.sum_feed(inputs)
        return self.activation(total)

    def __str__(self):
        return "{}{}".format(
            ", ".join(
                [
                    "weight{}: {}".format(i, weight)
                    for i, weight in enumerate(self.weights)
                ]
            ),
            "{}bias: {}".format(", " if len(self.weights) else "", self.bias),
        )


class Neuron(BaseNeuron):
    pass
