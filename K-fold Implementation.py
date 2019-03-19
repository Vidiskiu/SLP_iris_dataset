# Math functions library, http://www.numpy.org/
import numpy as np
# Graph plotting library, https://matplotlib.org/
import matplotlib.pyplot as plt

# Take target value and answer with label name, only for iris dataset
def label(encoding):
    if(encoding[0][0] == 0 and encoding[1][0] == 0):
        return "Setosa"
    elif(encoding[0][0] == 0 and encoding[1][0] == 1):
        return "Versicolor"
    elif(encoding[0][0] == 1 and encoding[1][0] == 0):
        return "Virginica"
    else:
        return "Not Recognized"

def listDifference(list_A, list_B):
    temp_list = [item for item in list_A if item not in list_B]
    final_list = []
    for segment in temp_list:
        final_list.extend(segment)
    return final_list

# SLP Class
class SLP:
    # Defines general attributes of the SLP
    def __init__(self, input, output, learning_rate):
        self.input = input
        self.output = output
        self.learning_rate = learning_rate

        self.accuracy = 0
        self.epochs_accuracy = []
        self.errors = np.zeros((self.output,1))
        self.epochs_error = []

        self.k_avg_acc = [0]*100
        self.k_avg_err = [0]*100

        self.val_accuracy = 0
        self.val_epochs_accuracy = []
        self.val_errors = np.zeros((self.output,1))
        self.val_epochs_error = []

        self.val_k_avg_acc = [0]*100
        self.val_k_avg_error = [0]*100

        # Define random values for the weights, normal distribution of mean = 0 and standard deviation of 1/sqrt(input)
        # self.wio = np.random.normal(0, pow(self.input, 0.5), [self.output, self.input])

        # To verify results with weights in excel file
        self.wio = [[-0.092096285,-0.238528171,-0.896053633,0.078595675,0.193810545],[-0.217029555,0.779709376,-0.051469509,-0.020827727,0.389959038]]
        self.initialLR = self.wio

        # Defining sigmoid as activation function
        self.activation_function = lambda x : (1 / (1 + np.exp(-1 * x)))

    # Training function
    def train(self, dataset, validata, epochs):
        self.nnReset()
        for e in range(epochs):
            self.accReset()
            self.errorReset()

            for record in dataset:
                # Split data by comma
                values = record.split(",")
                # Input data and skip first row since it is the features name
                inputs = np.asfarray(values[0:4])
                # Add 1 as constant to be multiplied to bias
                inputs = np.append(inputs, [1])
                
                # Create target array
                targets = np.zeros((self.output, 1))

                # Target nodes
                targets[0,0] = np.asfarray(values[5])
                targets[1,0] = np.asfarray(values[6])

                # Transform inputs into array, transformed for calculation purposes
                inputs = np.array(inputs, ndmin = 2).T
                target = np.array(targets, ndmin = 2).T

                # Preparation for inputs
                outputs_inputs = np.dot(self.wio, inputs)
                outputs_outputs = self.activation_function(outputs_inputs)

                # Errors
                outputs_errors = (targets - outputs_outputs) ** 2
                self.errors += outputs_errors

                # Updating weights
                self.wio -=  self.learning_rate * 2 * np.dot(((outputs_outputs - targets) * outputs_outputs * (1 - outputs_outputs)), inputs.T)

                # Guess using current SLP
                inputs = np.array(inputs, ndmin = 2).T
                query = self.query(inputs)

                # Rounding SLP outputs
                query[query >= 0.5] = 1
                query[query < 0.5] = 0

                if(label(query) == label(targets)):
                    self.accuracy += 1

            self.validate(validata)

            self.epochs_accuracy.append(self.accuracy / len(dataset))
            self.epochs_error.append(self.errors / len(dataset))
            self.val_epochs_accuracy.append(self.val_accuracy / len(validata))
            self.val_epochs_error.append(self.val_errors / len(validata))

        self.getStats()

    def validate(self, validata):
        for record in validata:
                # Split data by comma
                values = record.split(",")
                # Input data and skip first row since it is the features name
                inputs = np.asfarray(values[0:4])
                # Add 1 as constant to be multiplied to bias
                inputs = np.append(inputs, [1])
                
                # Create target array
                targets = np.zeros((self.output, 1))

                # Target nodes
                targets[0,0] = np.asfarray(values[5])
                targets[1,0] = np.asfarray(values[6])

                # Transform inputs into array, transformed for calculation purposes
                inputs = np.array(inputs, ndmin = 2).T
                target = np.array(targets, ndmin = 2).T

                # Preparation for inputs
                outputs_inputs = np.dot(self.wio, inputs)
                outputs_outputs = self.activation_function(outputs_inputs)

                # Errors
                outputs_errors = (targets - outputs_outputs) ** 2
                self.val_errors += outputs_errors

                # Guess using current SLP
                inputs = np.array(inputs, ndmin = 2).T
                query = self.query(inputs)
                # Rounding SLP outputs
                query[query >= 0.5] = 1
                query[query < 0.5] = 0

                if(label(query) == label(targets)):
                    self.val_accuracy += 1

    # Using current weights, produce outputs from inputs
    def query(self, inputs):
        inputs = np.array(inputs, ndmin = 2).T

        outputs_inputs = np.dot(self.wio, inputs)
        outputs_outputs = self.activation_function(outputs_inputs)

        return outputs_outputs

    def kFold(self, dataset, epochs, k):
        kdataset = []

        for i in range(k):
            kdataset.append([])
       
        counter = 0

        # Distribute dataset according to numbers of features
        for record in dataset:
            kdataset[counter % k].append(record)
            counter += 1

        for kval in range(k):
            dataset_val = []
            dataset_val.append(kdataset[kval])
            dataset_train = listDifference(kdataset, dataset_val)
            dataset_val = dataset_val[0]

            self.train(dataset_train, dataset_val, epochs)

        self.getStatsK(k)

    def getStats(self):
        # Preparing x values for graph plotting
        x = np.linspace(1, epochs, epochs)

        epochs_error = np.asfarray(self.getError())
        epochs_acc = np.asfarray(self.getAccuracy())
        val_epochs_error = np.asfarray(self.getValError())
        val_epochs_acc = np.asfarray(self.getValAccuracy())

        # Combined error
        epochs_error = epochs_error[:, 0] + epochs_error[:, 1]
        temp = []

        for e in epochs_error:
            temp.extend(e)

        epochs_error = temp

        # Combined val error
        val_epochs_error = val_epochs_error[:, 0] + val_epochs_error[:, 1]
        temp = []

        for e in val_epochs_error:
            temp.extend(e)

        val_epochs_error = temp

        print("Minimum Error :", round(min(epochs_error), 3))
        print("Average Error :", round(sum(epochs_error) / epochs, 3))
        print("Maxium Accuracy :", round(max(epochs_acc), 3))
        print("Average Accuracy :", round(sum(epochs_acc) / epochs, 3))
        print("Minimum Val Error :", round(min(val_epochs_error), 3))
        print("Average Val Error :", round(sum(val_epochs_error) / epochs, 3))

        self.k_avg_err = self.k_avg_err + np.asfarray(epochs_error)
        self.k_avg_acc = self.k_avg_acc + epochs_acc
        self.val_k_avg_err = self.val_k_avg_error + np.asfarray(val_epochs_error)
        self.val_k_avg_acc = self.val_k_avg_acc + val_epochs_acc

        plt.xlim(0,epochs)
        plt.ylim(0,1)
        plt.title("Accuracy Graph")
        plt.xlabel('Epoch(s)')
        plt.ylabel('Accuracy')
        plt.plot(x, val_epochs_acc)
        plt.plot(x, epochs_acc)
        plt.legend(["Val Acc", "Train Accuracy"], loc = "lower right")
        plt.show()
        
        plt.title("Error Graph")
        plt.xlabel('Epoch(s)')
        plt.ylabel('Error')
        plt.plot(x, val_epochs_error)
        plt.plot(x, epochs_error)
        plt.legend(["Val Err", "Train Error"], loc = "lower right")
        plt.show()

    def getStatsK(self, k):
        # Preparing x values for graph plotting
        x = np.linspace(1, epochs, epochs)

        self.k_avg_acc = self.k_avg_acc / k
        self.k_avg_err = self.k_avg_err / k

        self.val_k_avg_acc = self.val_k_avg_acc / k
        self.val_k_avg_err = self.val_k_avg_err / k

        print("Minimum K - Error :", round(float(min(self.k_avg_err)), 3))
        print("Average K - Error :", round(float(sum(self.k_avg_err / epochs)), 3))
        print("Maxium K - Accuracy :", round(max(self.k_avg_acc), 3))
        print("Average K - Accuracy :", round(sum(self.k_avg_acc) / epochs, 3))

        print("Minimum K - Error :", round(float(min(self.val_k_avg_err)), 3))
        print("Average K - Error :", round(float(sum(self.val_k_avg_err / epochs)), 3))
        print("Maxium K - Accuracy :", round(max(self.val_k_avg_acc), 3))
        print("Average K - Accuracy :", round(sum(self.val_k_avg_acc) / epochs, 3))

        plt.xlim(0,epochs)
        plt.ylim(0,1)
        plt.title("Average Accuracy Graph")
        plt.xlabel('Epoch(s)')
        plt.ylabel('Accuracy')
        plt.plot(x, self.val_k_avg_acc)
        plt.plot(x, self.k_avg_acc)
        plt.legend(["Val Acc", "Train Accuracy"], loc = "lower right")
        plt.show()
        
        plt.title("Average Error Graph")
        plt.xlabel('Epoch(s)')
        plt.ylabel('Error')
        plt.plot(x, self.val_k_avg_err)
        plt.plot(x, self.k_avg_err)
        plt.legend(["Val Err", "Train Error"], loc = "lower right")
        plt.show()

    # Get accuracy list
    def getAccuracy(self):
        return self.epochs_accuracy

    # Get error list
    def getError(self):
        return self.epochs_error

    # Get accuracy list
    def getValAccuracy(self):
        return self.val_epochs_accuracy

    # Get error list
    def getValError(self):
        return self.val_epochs_error

    # Clear variable error for next epoch's error calculation
    def errorReset(self):
        self.val_errors = np.zeros((self.output,1))
        self.errors = np.zeros((self.output,1))

    # Clear variable accuracy for next epoch's accuracy calculation
    def accReset(self):
        self.val_accuracy = 0
        self.accuracy = 0
    
    # Reset learning rate
    def nnReset(self):
        self.wio = self.initialLR

        self.epochs_error = []
        self.epochs_accuracy = []
        
        self.val_epochs_accuracy = []
        self.val_epochs_error = []

# Numbers of inputs, add 1 for bias
input_nodes = 4 + 1
# Numbers of outputs
output_nodes = 2
# Learning rate
learning_rate = 0.1
# Epochs
epochs = 100

# Creation of SLP instance
SLP_object = SLP(input_nodes, output_nodes, learning_rate)

# Dataset Preparation
train_data_file = open("iris.csv", "r")
training_data_list = train_data_file.readlines()
# Excluding the column names
training_data_list = training_data_list[1:]
train_data_file.close()

# SLP_object.train(training_data_list, epochs)
SLP_object.kFold(training_data_list, epochs, 5)