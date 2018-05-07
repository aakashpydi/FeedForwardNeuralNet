import numpy as np
import torch
import math

class NeuralNetwork:

    # Create the dictionary of matrixes Theta (layer). provides
    # theta weight value from layer to layer
    def __init__(self, network_layers):
        #network layers gives no. of nodes in layers,
        #input layer --> hidden layer_1 --> hidden layer_2 --> ... --> output layer

        #need to initialize theta dictionary
        self.network_layers = network_layers
        self.theta = dict()

        self.edges_keys = ["" for x in range(len(self.network_layers) - 1)]

        for i in range(len(self.network_layers) - 1):
            self.edges_keys[i] = "from:" +str(i)+"--"+"to:"+str(i+1)


        for i in range(len(self.network_layers) - 1):
            #need to initalize theta dictionary values. with mean 0 and std dev <- 1/sqrt(layer_size)
            nodes_in_current_layer = network_layers[i] + 1 #add bias node to each layer
                                                                    #Bias nodes are added to feedforward neural networks to help them learn patterns.
                                                                    #Bias nodes function like an input node that always produces constant. THe constant is called bias activation
            nodes_in_next_layer = network_layers[i + 1]             #note bias nodes have no input nodes
            size = (nodes_in_current_layer, nodes_in_next_layer)
            random = np.random.normal(0, 1/math.sqrt(self.network_layers[i]),size)

            self.theta.update({self.edges_keys[i]:torch.from_numpy(random)}) #update dictionary

    def getLayer(self, layer_requested):
        return self.theta[self.edges_keys[layer_requested]]


    def forward(self, input_tensor):
        tensor_passed_through_network = input_tensor
        #print input_tensor
        #print input_tensor.size()
        (x, y) = input_tensor.size()
        bias_node = torch.ones((1, y)) #returns tensor with dimension 1 * y
        bias_node = bias_node.type(torch.DoubleTensor) #in case input is a 1d DoubleTensor

        for i in range(len(self.network_layers) - 1):
            tensor_passed_through_network = torch.cat((bias_node, tensor_passed_through_network), dim=0) #concatenated along dimension 0
            layer_weights_transposed = torch.t(self.theta[self.edges_keys[i]])
            results = torch.mm(layer_weights_transposed, tensor_passed_through_network)

            # need to use sigmoid on results to see what values propagated forward
            results_np = results.numpy()
            results_np = 1/(1 + np.exp(-results_np))
            results = torch.from_numpy(results_np)
            tensor_passed_through_network = results

        return tensor_passed_through_network
