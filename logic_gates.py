from neural_network import NeuralNetwork
import torch
import numpy as np

class AND:
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        self.theta = self.gate.getLayer(0)
        self.theta.fill_(0)
        self.theta += torch.DoubleTensor([[-10], [6], [6]])

    def __call__(self, input_1, input_2):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0
        if input_2 == True:
            self.input_2 = 1
        if input_2 == False:
            self.input_2 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.input_1],[self.input_2]]))

class OR:
    def __init__(self):
        self.gate = NeuralNetwork([2, 1])
        self.theta = self.gate.getLayer(0)
        self.theta.fill_(0)
        self.theta += torch.DoubleTensor([[-1], [2], [2]])

    def __call__(self, input_1, input_2):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0
        if input_2 == True:
            self.input_2 = 1
        if input_2 == False:
            self.input_2 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.input_1],[self.input_2]]))

class NOT:
    def __init__(self):
        self.gate = NeuralNetwork([1, 1])
        self.theta = self.gate.getLayer(0)
        self.theta.fill_(0)
        #print self.theta
        self.theta += torch.DoubleTensor([[1], [-2]])

    def __call__(self, input_1):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.input_1]]))


class XOR:
    def __init__(self):
        self.gate = NeuralNetwork([2, 2, 1])
        self.theta_layer_0 = self.gate.getLayer(0)
        self.theta_layer_0.fill_(0)
        self.theta_layer_0 += torch.DoubleTensor([[-25, -25], [-50, 50], [50, -50]])

        self.theta_layer_1 = self.gate.getLayer(1)
        self.theta_layer_1.fill_(0)
        self.theta_layer_1 += torch.DoubleTensor([[-25], [50], [50]])

    def __call__(self, input_1, input_2):
        if input_1 == True:
            self.input_1 = 1
        if input_1 == False:
            self.input_1 = 0
        if input_2 == True:
            self.input_2 = 1
        if input_2 == False:
            self.input_2 = 0

        result = np.around(self.forward().numpy()) #round output of propagation to 0 or 1
        #print result
        if result == 1:
            return True
        else:
            return False

    def forward(self):
        return self.gate.forward(torch.DoubleTensor([[self.input_1],[self.input_2]]))
