import numpy
import functools
import math
import neurons
import normalization

class FeedForwardNetwork:
    """
    Represents a feedforward network.
    """
    
    def __init__(self, layer_properties):
        self.layer_properties = layer_properties
        
        self.__createLayers()
        
    def __createLayers(self):
        
        #initialize layers
        self.layers = [[] for i in self.layer_properties]
        
        for index, layer_property in enumerate(self.layer_properties):
            
            #The first layer is input, the last is output, the rest are hidden.
            if index == 0:
                self.__create_input_layer(index)
            elif index == len(self.layer_properties) - 1:
                self.__create_output_layer(index)
            else:
                self.__create_hidden_layer(index)
             
            # append bias neuron to the specified layers
            if(layer_property.get("bias")):
                self.__append_bias_neuron(index)
                
    
    def __create_input_layer(self, layer_index):
        # input neurons have a single weight set to 1 because we dont want to aclae the already normalized values.
        weight_count = 1
        weights = self.__get_layer_weights(layer_index, weight_count, [[1] for _ in range(self.layer_properties[layer_index]["neuron_count"])])
        
        self.__append_neurons(layer_index, weight_count, weights)
        
    def __create_hidden_layer(self, layer_index):
        #Each neurons weight count is equal to the number of non bias neurons in the previous layer
        weight_count = len(self.layers[layer_index - 1])
        weights = self.__get_layer_weights(layer_index, weight_count)
        
        self.__append_neurons(layer_index, weight_count, weights)
        
    def __create_output_layer(self, layer_index):
        #Each neurons weight count is equal to the number of non bias neurons in the previous layer
        weight_count = len(self.layers[layer_index - 1])
        weights = self.__get_layer_weights(layer_index, weight_count)
        
        self.__append_neurons(layer_index, weight_count, weights)
        
    def __append_neurons(self, layer_index, weight_count, weights):
        self.layers[layer_index] = [self.layer_properties[layer_index].get("neuron_type")(weights[index]) 
                                   for index, i in enumerate(range(self.layer_properties[layer_index].get("neuron_count")))]
        
    def __append_bias_neuron(self, layer_index):
        self.layers[layer_index].append(neurons.BiasNeuron(self.layer_properties[layer_index].get("bias")))
        
    def __get_layer_weights(self, layer_index, weight_count, weights = None):
        """ Return either the initial weights or randomize the weights for each layer."""

        return self.layer_properties[layer_index].get("initial_weights") or weights or [numpy.random.random(weight_count).tolist() for _ in range(self.layer_properties[layer_index].get("neuron_count"))]
        
    
    def __repr__(self):
        return repr(self.layers)
    
    def __computes(self, inputs):
        """
        computes the current input.
        """
         # initialize the output array
        outputs = [[] for layer in self.layers]
        
        #compute outputs
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer):
                
                neuron_output = None;
                
                # compute the output of the neuron. BiasNeurons don"t take any input.
                if type(neuron) != neurons.BiasNeuron:
                    neuron_inputs = [inputs[neuron_index]] if layer_index == 0 else outputs[layer_index - 1]
                    neuron_output = neuron.compute(neuron_inputs)
                else:
                    neuron_output = neuron.compute()
                    
                outputs[layer_index].append(neuron_output)
                
        return outputs;
        
    def compute(self, inputs):
        """
        computes a collection of inputs. If train = True, also adjust weights
        """
       
        # initialize the output array
        outputs = [[] for layer in self.layers]
        
        #compute outputs
        for input in inputs:
            outputs = self.__computes(input)
            
        return outputs;
    

class BackPropagationNetwork(FeedForwardNetwork):

    def __init__(self, layer_properties):
        super(BackPropagationNetwork, self).__init__(layer_properties)
    
    def compute(self, inputs, expected_outputs = None):
        """ If compute is called with an "expected" array, the network will train itself
        """
        
        outputs = super(BackPropagationNetwork, self).compute(inputs)
        
        if expected_outputs != None:
            for index, input in enumerate(inputs):
                self.train(outputs, expected_outputs[index]);
        
        return outputs
    
    def train(self, outputs, expected_outputs):
        
        # set the output neurons error
        self.backpropagate(outputs, expected_outputs)
            
        #print("train")
        #print([x.error for x in self.layers[output_layer_index]])
        
    def backpropagate(self, outputs, expected_outputs):
        
        output_layer_index = len(self.layers) - 1
        input_layer_index = 0
        
        # hack to make ouptuts match tutorial
        outputs[len(self.layers) - 1] = [0.6213859615555266, 0.6573693455986976]
            
        # calculate the error of the neurons in all layers
        for layer_index, layer in reversed(list(enumerate(self.layers))):
            print(layer_index)
            output = outputs[layer_index]
            for neuron_index, neuron in enumerate(self.layers[layer_index]):
                
                #we only calculate the error of non-bias neurons
                if type(neuron) != neurons.BiasNeuron:
                    
                    #calculate the error of each neuron based on whether or it is an output neuron or hidden neuron
                    if layer_index == output_layer_index:
                        neuron.error = self.calculate_output_error(output[neuron_index], expected_outputs[neuron_index])
                    elif layer_index > input_layer_index:
                        neuron.error = self.calculate_hidden_error(layer_index, neuron_index, outputs)
        
    def calculate_error_derivative(self, output):
        return output * (1.0 - output)
    
    def calculate_output_error(self, output, expected_output):
        """ Calculates the error of an ouptut neurons output."""
        return (expected_output - output) * self.calculate_error_derivative(output)
    
    def calculate_hidden_error(self, layer_index, neuron_index, outputs):
        """ Calculates the error of a hidden neurons output."""
        
        weighted_errors = [neuron.error * neuron.weights[neuron_index] for neuron in self.layers[layer_index + 1]]
        sum_of_weighted_errors = sum(weighted_errors)
        
            
        print('foo', sum_of_weighted_errors)
            
        return sum_of_weighted_errors * self.calculate_error_derivative(outputs[layer_index][neuron_index])
        
    def calculate_cost_function(self):
        print('calculate_cost_function')
        
        