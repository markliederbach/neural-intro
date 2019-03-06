import numpy as np
from network.neurons import Neuron

class NeuralNetwork:

    def __init__(self, hidden_layer_neurons, num_inputs):
        self.hidden_neurons = [Neuron(num_inputs) for _ in range(hidden_layer_neurons)]
        self.output_neuron = Neuron(hidden_layer_neurons)

    @staticmethod
    def _to_npa(l):
        return l if isinstance(l, np.ndarray) else np.array(l)

    @staticmethod
    def mse_loss(answers, predictions):
        return ((answers - predictions) ** 2).mean()

    def feedforward(self, inputs):
        hidden_results = [h.feedforward(self._to_npa(inputs)) for h in self.hidden_neurons]
        return self.output_neuron.feedforward(self._to_npa(hidden_results))

    def train(self, training_set, answers, learn_rate=None, epochs=None):
        learn_rate = 0.1 if learn_rate is None else learn_rate
        epochs = 1000 if epochs is None else epochs

        for epoch in range(epochs):
            for item, answer in zip(training_set, answers):
                h_results = []
                for index in range(0, len(self.hidden_neurons)):
                    h_sum = self.hidden_neurons[index].sum_feed(self._to_npa(item))
                    h_activation = self.hidden_neurons[index].activation(h_sum)
                    h_results.append({'sum': h_sum, 'activation': h_activation})

                o_sum = self.output_neuron.sum_feed(self._to_npa([h['activation'] for h in h_results]))

                prediction = self.output_neuron.activation(o_sum)


                # Calculate partial deriv
                dL_dprediction = -2 * (answer - prediction)

                # Calculate partial derivs on neurons based on each weights
                dprediction_dw1 = item[0] * self.h1.deriv_activation(h1_sum)
                dprediction_dw2 = item[1] * self.h1.deriv_activation(h1_sum)
                dprediction_dbias1 = self.h1.deriv_activation(h1_sum)

                dprediction_dw3 = item[0] * self.h2.deriv_activation(h2_sum)
                dprediction_dw4 = item[1] * self.h2.deriv_activation(h2_sum)
                dprediction_dbias2 = self.h2.deriv_activation(h2_sum)

                dprediction_dw5 = h1_activation * self.o1.deriv_activation(o1_sum)
                dprediction_dw6 = h2_activation * self.o1.deriv_activation(o1_sum)
                dprediction_dbias3 = self.o1.deriv_activation(o1_sum)
                dprediction_dh1 = self.o1.weights[0] * self.o1.deriv_activation(o1_sum)
                dprediction_dh2 = self.o1.weights[1] * self.o1.deriv_activation(o1_sum)

                # Update weights and biases
                self.h1.update_weights(
                    self.h1.weights[0] - (learn_rate * dL_dprediction * dprediction_dh1 * dprediction_dw1),
                    self.h1.weights[1] - (learn_rate * dL_dprediction * dprediction_dh1 * dprediction_dw2),
                )
                self.h1.bias = self.h1.bias - (learn_rate * dL_dprediction * dprediction_dh1 * dprediction_dbias1)

                self.h2.update_weights(
                    self.h2.weights[0] - (learn_rate * dL_dprediction * dprediction_dh2 * dprediction_dw3),
                    self.h2.weights[1] - (learn_rate * dL_dprediction * dprediction_dh2 * dprediction_dw4),
                )
                self.h2.bias = self.h2.bias - (learn_rate * dL_dprediction * dprediction_dh2 * dprediction_dbias2)

                self.o1.update_weights(
                    self.o1.weights[0] - (learn_rate * dL_dprediction * dprediction_dw5),
                    self.o1.weights[1] - (learn_rate * dL_dprediction * dprediction_dw6),
                )
                self.o1.bias = self.o1.bias - (learn_rate * dL_dprediction * dprediction_dbias3)

