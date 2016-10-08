import math
import functools
import numpy
import random

class LinearActivationFunction:
    '''
    Linear activation functions do not change the value of the input
    They are typically used in the output layer of a network learning
    Numerical Values.
    '''
    
    def compute(self, x):
        y = x
        return y
    
    
    
    
class StepActivationFunction:
    '''
    Step activation functions output a zero if the input is below a certain 
    threshold and a 1 if the input is at or above that threshold.
    '''
    
    def compute(self, x):
        y = 1 if x >= .5 else 0
        return y
    
    
    
class SigmoidActivationFunction:
    '''
    Sigmoid activation functions output a only positive numbers within a very narow range.
    '''
    def compute(self, x):
        y = 1 / (1 +math.exp(-x))
        return y
    

    
class HyperbolicTangentActivationFunction:
    '''
    Hyperbolic tangent activation functions output a very narow range between -1 and 1.
    '''
    def compute(self, x):
        y = math.tanh(x)
        return y
    
    

class RectifiedLinearUnitActivationFunction:
    '''
    Rectified Linear Unit activation functions scale linearly above a minimum threshold.
    '''
    def compute(self, x):
        y = max(0, x)

        return y


    
class SoftmaxActivationFunction:
    '''
    Softmax activation functions output a vector containing probabilities that each neuron in an output layer 
    is a member of the class represented by the Softmax function.
    '''
    
    def compute(self, output_neurons):
        #sum the exponent of each ouptut neuron value
        output_neuron_sum = sum([math.exp(x) for x in output_neurons])

        #Get probabilities by dividing the exponent of each output newron with the sum of the exponent of 
        #all output neurons.
        probabilities = [math.exp(x) / output_neuron_sum for x in output_neurons]

        return probabilities
    
class HopefieldActivationFunction:
    '''
    Hopefield activation functions calulate a state of +1 or -1 based on the sum of the product of the neurons weights
    and the states of the other neurons in the network.
    '''
    def compute(self, weights, pattern, threshold):
        length = len(weights)
        
        #find the state/weight dot product
        product = [weights[i] * pattern[i]
                  for i in range(length)]
        
        sum_of_product = numpy.sum(product)
        state = 1 if (sum_of_product > threshold) else 0
        return state
    
class Neuron(object):
    '''
    A base neuron that defines the weight and bias
    '''
    
    def __init__(self, weights, bias):
        #a vector contining a weight value for each input
        self.weights = weights
        self.bias = bias
        
    def compute_inputs(self, inputs):
        #calculate a new vector that is the dot_product of the inputs and weights
        dot_product = [input * self.weights[index] for index, input in  enumerate(inputs)]
        sum_of_dot_product = functools.reduce(lambda x,y : x + y, dot_product )
        return sum_of_dot_product + self.bias
    
    def __repr__(self):
        '''
        print neuron weights
        '''
        return repr(self.weights)


    
class StepActivationNeuron(Neuron):
    '''
    A neuron that uses a step activation function.
    '''
    
    def __init__(self, weights, bias):
        super(StepActivationNeuron, self).__init__(weights, bias)
        self.activation = StepActivationFunction()
    
    def compute(self, inputs):
        return self.activation.compute(self.compute_inputs(inputs))
    
       


class SigmoidActivationNeuron(Neuron):
    '''
    A neuron that uses a sigmoid activation function.
    '''
    
    def __init__(self, weights, bias):
        super(SigmoidActivationNeuron, self).__init__(weights, bias)
        self.activation = SigmoidActivationFunction()
    
    def compute(self, inputs):
        return self.activation.compute(self.compute_inputs(inputs))
    
class HopefieldActivationNeuron(Neuron):
    '''
    A neuron that uses a sigmoid activation function.
    '''
    
    def __init__(self, weights, bias):
        super(HopefieldActivationNeuron, self).__init__(weights, bias)
        self.activation = HopefieldActivationFunction()
        self.threshold = 0
        self.state = 0
    
    def compute(self, pattern):
        self.state = self.activation.compute(self.weights, pattern, self.threshold)
        return self.state
    
class BiasNeuron(Neuron):
    '''
    A bias neuron.
    '''
    
    def __init__(self, bias):
        '''
        Bias eurons are instantiated with a bias. Usually 1.
        '''
        super(BiasNeuron, self).__init__([], bias)
    
    def compute(self, pattern):
        return self.bias
    
    def __repr__(self):
        '''
        print bias
        '''
        return repr(self.bias)
    

