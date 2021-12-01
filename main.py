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

    # function to get Actual_Output given inputs x
    # input: list:x -> list that contains inputs (longitude, latitude)
    def predict(self, x):
        actual = []
        for i in range(self.output_num)
            actual.append(self.output(x, self.weights[i]))
        # for w in weights:
        #     actual.append(output(x, w))

        return actual

if __name__ == '__main__':
    print("Hello World!")