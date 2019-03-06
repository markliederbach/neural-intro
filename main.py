"""https://victorzhou.com/blog/intro-to-neural-networks/"""
import numpy


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + numpy.exp(-value))

    @classmethod
    def activation(cls, value):
        return cls.sigmoid(value)

    @classmethod
    def deriv_activation(cls, value):
        fx = cls.activation(value)
        return fx * (1 - fx)

    def sum_feed(self, inputs):
        return numpy.dot(self.weights, inputs) + self.bias

    def feedforward(self, inputs):
        total = self.sum_feed(inputs)
        return self.activation(total)

    def __str__(self):
        return "weight1: {}, weight2: {}, bias: {}".format(self.weights[0], self.weights[1], self.bias)


# weights = numpy.array([0, 1])
# bias = 4
#
# n = Neuron(weights, bias)
#
# inputs = numpy.array([2, 3])
# print(n.feedforward(inputs))


# class NeuralNetwork:
#     def __init__(self):
#         weights = numpy.array([0, 1])
#         bias = 0
#
#         # Define hidden layer
#         self.h1 = Neuron(weights, bias)
#         self.h2 = Neuron(weights, bias)
#
#         # Define output layer
#         self.o1 = Neuron(weights, bias)
#
#     def feedforward(self, inputs):
#         out_h1 = self.h1.feedforward(inputs)
#         out_h2 = self.h2.feedforward(inputs)
#
#         # Inputs for output layer are outputs from hidden layer
#         out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))
#
#         return out_o1


# net = NeuralNetwork()
# inputs = numpy.array([2, 3])
# print(net.feedforward(inputs))


class NeuralNetwork:
    def __init__(self):

        # Define hidden layer
        self.h1 = Neuron(numpy.array([numpy.random.normal(), numpy.random.normal()]), numpy.random.normal())
        self.h2 = Neuron(numpy.array([numpy.random.normal(), numpy.random.normal()]), numpy.random.normal())

        # Define output layer
        self.o1 = Neuron(numpy.array([numpy.random.normal(), numpy.random.normal()]), numpy.random.normal())

    def gender(self, inputs):
        result = self.feedforward(inputs)

        if result <= 0.25:
            return "Male ({0:.2f}%)".format((1-result)*100)
        if result >= 0.75:
            return "Female ({0:.2f}%)".format(result*100)
        return "Unsure (value: {0:.3f})".format(result)

    def feedforward(self, inputs):
        out_h1 = self.h1.feedforward(inputs)
        out_h2 = self.h2.feedforward(inputs)

        # Inputs for output layer are outputs from hidden layer
        out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))

        return out_o1

    @staticmethod
    def mse_loss(answers, predictions):
        return ((answers - predictions) ** 2).mean()

    def train(self, data, answers):

        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for person, answer in zip(data, answers):

                # Feedforward each neuron             # height   # weight
                h1_sum = self.h1.sum_feed(person)
                h1_activation = self.h1.activation(h1_sum)

                h2_sum = self.h2.sum_feed(person)
                h2_activation = self.h2.activation(h2_sum)

                o1_sum = self.o1.sum_feed(numpy.array([h1_activation, h2_activation]))
                o1_activation = self.o1.activation(o1_sum)

                prediction = o1_activation

                # Calculate partial deriv
                dL_dprediction = -2 * (answer - prediction)

                # Calculate partial derivs on neurons based on each weights
                dprediction_dw1 = person[0] * self.h1.deriv_activation(h1_sum)
                dprediction_dw2 = person[1] * self.h1.deriv_activation(h1_sum)
                dprediction_dbias1 = self.h1.deriv_activation(h1_sum)

                dprediction_dw3 = person[0] * self.h2.deriv_activation(h2_sum)
                dprediction_dw4 = person[1] * self.h2.deriv_activation(h2_sum)
                dprediction_dbias2 = self.h2.deriv_activation(h2_sum)

                dprediction_dw5 = h1_activation * self.o1.deriv_activation(o1_sum)
                dprediction_dw6 = h2_activation * self.o1.deriv_activation(o1_sum)
                dprediction_dbias3 = self.o1.deriv_activation(o1_sum)
                dprediction_dh1 = self.o1.weights[0] * self.o1.deriv_activation(o1_sum)
                dprediction_dh2 = self.o1.weights[1] * self.o1.deriv_activation(o1_sum)

                # Update weights and biases
                self.h1.weights = numpy.array(
                    [
                        self.h1.weights[0] - (learn_rate * dL_dprediction * dprediction_dh1 * dprediction_dw1),
                        self.h1.weights[1] - (learn_rate * dL_dprediction * dprediction_dh1 * dprediction_dw2),
                    ]
                )
                self.h1.bias = self.h1.bias - (learn_rate * dL_dprediction * dprediction_dh1 * dprediction_dbias1)

                self.h2.weights = numpy.array(
                    [
                        self.h2.weights[0] - (learn_rate * dL_dprediction * dprediction_dh2 * dprediction_dw3),
                        self.h2.weights[1] - (learn_rate * dL_dprediction * dprediction_dh2 * dprediction_dw4),
                    ]
                )
                self.h2.bias = self.h2.bias - (learn_rate * dL_dprediction * dprediction_dh2 * dprediction_dbias2)

                self.o1.weights = numpy.array(
                    [
                        self.o1.weights[0] - (learn_rate * dL_dprediction * dprediction_dw5),
                        self.o1.weights[1] - (learn_rate * dL_dprediction * dprediction_dw6),
                    ]
                )
                self.o1.bias = self.o1.bias - (learn_rate * dL_dprediction * dprediction_dbias3)
                # Every 10 iterations, print status
            # if epoch % 10 == 0:
            #     predictions = numpy.apply_along_axis(self.feedforward, 1, data)
            #     loss = self.mse_loss(answers, predictions)
            #     print("Epoch {0}: Loss={1:.3f}".format(epoch, loss))


train_data = numpy.array([
  [-2, -1],   # Alice
  [25, 6],    # Bob
  [17, 4],    # Charlie
  [-15, -6],  # Diana
])
all_answers = numpy.array([
  1,  # Alice
  0,  # Bob
  0,  # Charlie
  1,  # Diana
])

network = NeuralNetwork()
network.train(train_data, all_answers)

emily = numpy.array([-7, -3])  # 128 pounds, 63 inches
frank = numpy.array([20, 2])  # 155 pounds, 68 inches
print("Emily: {}".format(network.gender(emily)))  # Female
print("Frank: {}".format(network.gender(frank)))  # Male
