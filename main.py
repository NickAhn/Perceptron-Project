import numpy as np

class Perceptron:
    def __init__(self, input_num, output_num):
        self.weights = []
        self.input_num = input_num
        self.output_num = output_num

    def output(self, x, weights):
        z = np.dot(x, weights)
        if z>0.5: return 1
        else: return 0

if __name__ == '__main__':
    print("Hello World!")