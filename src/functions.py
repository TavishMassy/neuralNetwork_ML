import numpy as np

class Functional:
    
    @staticmethod
    def ReLU(x):
        """
        Rectified Linear Unit.
        If a number is less than 0, it becomes 0. Otherwise, it stays the same.
        """
        # np.maximum compares every element with 0 and keeps the higher one
        return np.maximum(0, x)

    @staticmethod
    def Sigmoid(x):
        """
        Squishes every number in the matrix into a decimal between 0 and 1.
        Perfect for binary classification.
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def Tanh(x):
        """
        Squishes every number between -1 and 1. 
        """
        # Writing out the raw math formula for Tanh!
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def Softmax(x):
        """
        Turns an array of raw scores into probabilities that all add up to exactly 1.0.
        """
        # Subtract the max value for numerical stability
        # In NumPy, 'dim' is called 'axis'
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        
        exp_x = np.exp(shifted_x)
        
        # Divide each exponent by the sum of all exponents
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Creating the standard 'F' alias
F = Functional()