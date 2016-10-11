import numpy
import functools
import math

def z_score_normalize(sequence):
    '''
    Z-Score normalization returns a sequence normalized to the number of standard deviations from the mean.
    '''
    # Calculate mean.
    mean = numpy.sum(sequence) / len(sequence)
    
    # Calculate standard devieation
    standard_deviation =  math.sqrt(numpy.sum([(x - mean)**2 for x in sequence]) / len(sequence))
    
    # Calculate z scores
    normalized_sequence = [(x - mean) / standard_deviation for x in sequence]
    
    normalized_tuple = tuple(normalized_sequence)
    
    return normalized_tuple
    
    
def one_hot(sequence):
    '''
    One Hot nortmalization encodes unordered categorical data into numerical vectors that can be used by a neural net.
    '''
    sequence_length = len(sequence)
    encoded_sequence = [tuple(x) for x in numpy.eye(sequence_length)]
    encoded_tuple = tuple(encoded_sequence)
    return encoded_tuple