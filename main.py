import numpy as np

np.random.seed(0)

##
# CHANGE VALUES BELOW:
# #
LEARNING_RATE = 0.7
THRESHOLD_RATE = 0.06
EPOCHS = 10
TRAINING_DATA = "nnTrainData.txt"
TEST_DATA = "nnTestData.txt"
###

# perfectly classified output percentage:
# alpha = 0.7, threhsold=0.06, epochs=10: 91.48306451612903%
# alpha = 0.7, threhsold=0.06, epochs=20: 90.37717741935484%
# alpha = 0.7, threhsold=0.06, epochs=30: 91.89120967741935%
# alpha = 0.7, threhsold=0.06, epochs=70: 92.50588709677419%

def readFile(file_name):
    try:
        file = open(file_name, "r")
    except OSError:
        print("- Could not open/read file: " + file_name)
        return None

    coords =[]
    regions =[]
    for line in file:
        temp = line.split()
        latitude = (float(temp[0])+90)/180
        longitude = (float(temp[1]) + 180)/360
        coords.append((latitude,longitude))
        regions.append(getExpectedOutput(temp[2]))

    return coords,regions


def getExpectedOutput(region):
    dic = {
        "Africa":0,
        "America":1,
        "Antartica":2,
        "Asia":3,
        "Australia":4,
        "Europe":5,
        "Arctic":6,
        "Atlantic":7,
        "Indian":8,
        "Pacific":9
    }
    index = dic.get(region)
    expected_output = []
    for i in range(10):
        if i == index:
            output = 1
        else:
            output = 0
        expected_output.append(output)

    return expected_output


class Perceptron:
    def __init__(self, input_num, output_num, learning_rate, threshold_rate, epoch_num):
        self.weights = []
        # num of weight pairs = num of outputs
        for i in range(output_num):
            weight = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
            self.weights.append(weight)

        self.input_num = input_num
        self.output_num = output_num
        self.alpha = learning_rate
        self.epoch_num = epoch_num
        self.threshold = threshold_rate
        self.regions = ["Africa", "America", "Antarctica", "Asia", "Australia", "Europe", "Arctic", "Atlantic", "Indian", "Pacific"]

    # Creates the output of a single Neuron of the Perceptron
    def output(self, x, weights):
        z = np.dot(x, weights)
        if z>self.threshold: return 1
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
    def train(self, X, expected_outputs):
        # iterating thru each sample\
        print("------------Start of Training------------")
        for epoch in range(self.epoch_num):
            print("- EPOCH " + str(epoch) + " -")
            for x, exp in zip(X, expected_outputs):
                actual = self.predict(x)

                # Compare Actual Outputs with Expected Outputs #
                for i in range(self.output_num):
                    # print("\n--> Neuron " + str(i))

                    # Adjust Weights #
                    for j in range(self.input_num):
                        self.weights[i][j] = self.weights[i][j] + self.alpha*(exp[i] - actual[i])*x[j]

        print("---------End of training------------")
        self.printWeights()

    # function to write final weights after training in a separate text file
    def printWeights(self):
        file = open("trainedWeights.txt", 'w')
        for region, w in zip(self.regions, self.weights):
            file.write(region + " " + str(w[0]) + " " + str(w[1]) + "\n")

        file.close()

    def print_neuron_stats(self, arr, data_count):
        for i in range(len(self.regions)):
            print("Neuron: ", self.regions[i])
            print("\tCorrect: ", (arr[i][0] / data_count) * 100, "%")
            print("\tTrue Positives: ", (arr[i][1] / data_count) * 100, "%")
            print("\tTrue Negatives: ", (arr[i][2] / data_count) * 100, "%")
            print("\tFalse Positives: ", (arr[i][3] / data_count) * 100, "%")
            print("\tFalse Negatives: ", (arr[i][4] / data_count) * 100, "%")

    def run(self, X, expected_outputs):
        neuron_stats = [[0 for i in range(5)] for j in range(10)]
        print(neuron_stats)

        print("\n---------Start of Testing------------")
        perfect_count = 0

        data_count = 0 # variable to count the total number of sample
        none_fired_count = 0
        multiple_fired_count = 0
        # go through each sample and compare actual vs expected output
        for x, exp in zip(X, expected_outputs):
            actual = self.predict(x)
            neurons_fired_count = 0 # Count how many outputs were = 1

            # go through each output
            for i in range(self.output_num):
                if actual[i] == exp[i]:
                    #correct
                    perfect_count += 1

                if actual[i] == 1:
                    neurons_fired_count += 1

                #Counting Individual Neurons
                if actual[i] == exp[i]:
                    # Correct
                    neuron_stats[i][0] += 1
                if actual[i] == 1 and exp[i] == 1:
                    # True Positive
                    neuron_stats[i][1] += 1
                elif actual[i] == 0 and exp[i] == 0:
                    # True Negative
                    neuron_stats[i][2] += 1
                elif actual[i] == 1 and exp[i] == 0:
                    # False Positive
                    neuron_stats[i][3] += 1
                else:
                    #False Negative
                    neuron_stats[i][4] += 1

            data_count += 1

            if neurons_fired_count > 1:
                multiple_fired_count += 1
            elif multiple_fired_count == 0:
                none_fired_count += 1

        print("\n---------End of Testing------------")
        # print("- Number of perfectly classified outputs: " + str(float(perfect_count) / self.output_num))
        # print("- data count: " + str(data_count))
        print("- Percentage of the examples in the testing data set were perfectly classified: "
              + str((perfect_count/self.output_num)*100/data_count) + "%")

        print("- Percentage of the examples in the testing data set caused multiple neurons to fire: "
              + str(multiple_fired_count*100/data_count) + "%")
        print("- Percentage of the examples in the testing data set caused zero neurons to fire: "
              + str(none_fired_count*100/data_count) + "%")

        self.print_neuron_stats(neuron_stats, data_count)

if __name__ == '__main__':
    coords_train, expected_output_train = readFile(TRAINING_DATA)

    perceptron = Perceptron(2, 10, LEARNING_RATE, THRESHOLD_RATE, EPOCHS)
    perceptron.train(coords_train, expected_output_train)

    coords_test, expected_output_test = readFile(TEST_DATA)
    perceptron.run(coords_test, expected_output_test)




