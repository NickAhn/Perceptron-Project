import numpy as np

class Perceptron:
    def __init__(self, input_num, output_num):
        self.weights = []
        # num of weight pairs = num of outputs
        for i in range(output_num):
            # TODO: make it random
            weight = [0, 0]
            self.weights.append(weight)

        self.input_num = input_num
        self.output_num = output_num
        self.alpha = 1

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

            print("\n---------------TRAINING " + str(x))
            print("EXPECTED OUTPUT: " + str(exp))
            print("ACTUAL OUTPUT: " + str(actual))

            for i in range(self.output_num):
                print("\n--> Neuron " + str(i))
                for j in range(self.input_num):
                    print(" Weight before = " + str(self.weights[i][j]))
                    print(" " + str(self.weights[i][j]) + " = " + str(self.weights[i][j]) + " + (" + str(
                        exp[i]) + " - " + str(actual[i]) + ")*" + str(x[j]))

                    self.weights[i][j] = self.weights[i][j] + self.alpha*(exp[i] - actual[i])*x[j]
                    print(" Weight After = " + str(self.weights[i][j]) + "\n")


X = [[2, 1],
     [3, 3]]

expected_output_test = [[0, 1],
                        [1, 0]]

if __name__ == '__main__':
    perceptron = Perceptron(2, 2)
    perceptron.train(X, expected_output_test)