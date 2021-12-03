import numpy as np

np.random.seed(0)

##
# CHANGE VALUES BELOW:
# #
LEARNING_RATE = 0.7
THRESHOLD_RATE = 0.06
EPOCHS = 70
TRAINING_DATA = "nnTrainData.txt"
TEST_DATA = "nnTestData.txt"
###

'''
region_dict = {
        "Africa", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "America", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "Antartica", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Asia", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "Australia", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Europe", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "Arctic", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "Atlantic", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "Indian", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Pacific", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
'''

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
    index = 0  # Africa
    if (region == "America"):
        index = 1
    elif (region == "Antartica"):
        index = 2
    elif (region == "Asia"):
        index = 3
    elif (region == "Australia"):
        index = 4
    elif (region == "Europe"):
        index = 5
    elif (region == "Arctic"):
        index = 6
    elif (region == "Atlantic"):
        index = 7
    elif (region == "Indian"):
        index = 8
    elif (region == "Pacific"):
        index = 9
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
            # TODO: make it random
            weight = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
            # weight = [np.random.uniform(10.5, 75.5), np.random.uniform(10.5, 75.5)]
            self.weights.append(weight)

        self.input_num = input_num
        self.output_num = output_num
        self.alpha = learning_rate
        self.epoch_num = epoch_num
        self.threshold = threshold_rate

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

    def printWeights(self):
        file = open("trainedWeights.txt", 'w')
        counter = 0

        print("Final Weights:")
        for w in self.weights:
            for i in range(len(w)):
                # print(" w" + str(counter) + ": " + str(w[i]))
                file.write("w" + str(counter) + ": " + str(w[i]) + "\n")
                counter += 1
        file.close()

    def run(self, X, expected_outputs):
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
                    perfect_count += 1

                if actual[i] == 1:
                    neurons_fired_count += 1

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


    '''
    - What percentage of the examples in the testing data set were perfectly classified:  i.e. only 
    the correct neuron and none of the others fired.
    - What percentage of the examples in the testing data set caused multiple neurons to fire. 
    - What percentage of the examples in the testing data set caused zero neurons to fire. 
    '''

if __name__ == '__main__':
    coords_train, expected_output_train = readFile(TRAINING_DATA)

    perceptron = Perceptron(2, 10, LEARNING_RATE, THRESHOLD_RATE, EPOCHS)
    perceptron.train(coords_train, expected_output_train)

    coords_test, expected_output_test = readFile(TEST_DATA)
    perceptron.run(coords_test, expected_output_test)



# perfectly classified output percentage:
# alpha = 0.7, threhsold=0.06, epochs=10: 91.48306451612903%
# alpha = 0.7, threhsold=0.06, epochs=20: 90.37717741935484%
# alpha = 0.7, threhsold=0.06, epochs=30: 91.89120967741935%
#Perceptron(2, 10, 0.7, 0.06, 70) = 92.50588709677419% <- best
