import numpy
import functools
import math


def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
            
def normalize(types, data):
    """ Returns a fully normalized data set.
    
    Args:
        normalizer_types: A collection of normalizer types for each column in the data
        data: The full data set to be normalized
    """
    
    normalized_data = [ normalizer([x[index] for x in data]).data for index, normalizer in enumerate(types)]
    transposed_data = [tuple(flatten([r[col] for r in normalized_data])) for col in range(len(normalized_data[0]))]
    
    return transposed_data

class Normalizer:
    
    def __init__(self, training_data = ()):
        self.training_data = training_data
        self.data = None


class ZScore(Normalizer):
    """ Z-Score normalization returns a tuple of values normalized to the number of standard deviations from the mean. """
    
    def __init__(self, training_data = ()):
        super(ZScore, self).__init__(training_data)
        self.mean = None
        self.standard_deviation = None
            
        self.data = self.normalize(training_data)
        
    def normalize(self, data):
        """ Normalize the passed data.
        
        Args:
            data: A numeric list or tuple matrix of data to be normalized.
        
        Returns:
            A tuple matrix of notmalized data.
            
        """
        
        # Calculate mean if it isn't set
        if self.mean == None:
            self.mean = numpy.sum(self.training_data) / len(self.training_data)

        # Calculate standard deviation if it isn't set
        if self.standard_deviation == None:
            mean_difference_squared = [(x - self.mean)**2 for x in self.training_data]
            self.standard_deviation = math.sqrt(numpy.sum(mean_difference_squared) / len(self.training_data))

        # Calculate z scores
        normalized_data = [(x - self.mean) / self.standard_deviation for x in data]

        normalized_tuple = tuple(normalized_data)
        
        return normalized_tuple
    

class OneHot(Normalizer):
    """ One Hot normalization encodes unordered categorical data into numerical vectors that can be used by a neural net. """
    
    def __init__(self, training_data = ()):
        super(OneHot, self).__init__(training_data)
        
        # Because we are encoding categorical data we want to convert to a set to get a collection of unique values
        self.unique_training_data = tuple(set(training_data))
        self.categories = None
        self.categories = self.normalize(self.unique_training_data)
        self.data = self.normalize(training_data)
        
    
    def normalize(self, data):
        """ Normalize the passed data.
        
        Args:
            data: A categorical list or tuple matrix of data to be normalized.
        
        Returns:
            A tuple matrix of notmalized data.
            
        """
        
        if self.categories == None:
            data_length = len(data)
            encoded_data = [tuple(x) for x in numpy.eye(data_length)]
        else:
            encoded_data = [ self.categories[self.unique_training_data.index(x)] for x in data ]
        
        encoded_tuple = tuple(encoded_data)
        
        return encoded_tuple
    