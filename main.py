import numpy as np

class Perceptron:
    def __init__(self, input_num, output_num):
        self.weights = []
        self.input_num = input_num
        self.output_num = output_num

    # Creates the output of a single Neuron of the Perceptron
    def output(self, x, weights):
        z = np.dot(x, weights)
        if z>0.5: return 1
        else: return 0

    # Function to get Actual_Output given inputs x
    # input: list:x -> list that contains inputs (longitude, latitude)
    # output
    def predict(self, x):
        actual = []
        for i in range(self.output_num):
            actual.append(self.output(x, self.weights[i]))
        # for w in weights:
        #     actual.append(output(x, w))

        return actual

    # Train data
    # input:
    # - X-> list of samples. Each sample is a list with 2 inputs (longitude, latitude)
    # - expected_outputs -> list containing the expected outputs of each sample in X
    # - epoch_num = number of epocs before terminating the funciton
    # Output: adjust weights to improve accuracy of Perceptron
    def train(self, X, expected_outputs, epochs=10):
        #iterating thru each sample
        for x, exp in zip(X, expected_outputs):
            actual = self.predict(x)

            for i in range(self.output_num):
                for j in range(self.input_num):
                    #adjust weights
                    self.weights[i][j] = self.weights[i][j] + self.alpha*(exp[i] - actual[i])*input[j]


if __name__ == '__main__':
    print("Hello World!")