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
        return "Not Recognize"

# SLP Class
class SLP:
    # Defines general attributes of the SLP
    def __init__(self, input, output, learning_rate):
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.errors = np.zeros((2,1))
        self.accuracy = 0

        # Define random values for the weights, normal distribution of mean = 0 and standard deviation of 1/sqrt(input)
        # self.wio = np.random.normal(0, pow(self.input, 0.5), [self.output, self.input])
        # verify results with weights in excel file
        self.wio = [[-0.092096285,-0.238528171,-0.896053633,0.078595675,0.193810545],[-0.217029555,0.779709376,-0.051469509,-0.020827727,0.389959038]]

        # Defining sigmoid as activation function
        self.activation_function = lambda x : (1 / (1 + np.exp(-1 * x)))

    # Training function
    def train(self, inputs, targets):
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

    # Using current weights, produce outputs from inputs
    def query(self, inputs):
        inputs = np.array(inputs, ndmin = 2).T

        outputs_inputs = np.dot(self.wio, inputs)
        outputs_outputs = self.activation_function(outputs_inputs)

        return outputs_outputs

    # Clear variable error for next epoch's error calculation
    def errorReset(self):
        self.errors = np.zeros((2,1))

    def accReset(self):
        self.accuracy = 0

# numbers of inputs
input_nodes = 4 + 1
# numbers of outputs
output_nodes = 2
# learning rate
learning_rate = 0.1
# epochs
epochs = 100

SLP_object = SLP(input_nodes, output_nodes, learning_rate)
epochs_error = []
epochs_acc = []

train_data_file = open("iris.csv", "r")
training_data_list = train_data_file.readlines()
train_data_file.close()

for e in range(epochs):
    # Reset error
    SLP_object.errorReset()
    SLP_object.accReset()

    # Training for each records
    for record in training_data_list[1:]:
        # Split data by comma
        values = record.split(",")
        # Input data and skip first row since it is the features name
        inputs = np.asfarray(values[0:4])
        inputs = np.append(inputs,[1])
        # Create target array
        targets = np.zeros((output_nodes,1)) + 0.01
        # Target nodes
        targets[0,0] = np.asfarray(values[5])
        targets[1,0] = np.asfarray(values[6])
        # Train SLP
        query = SLP_object.query(inputs)
        query[query >= 0.5] = 1
        query[query < 0.5] = 0
        # Guess using current SLP
        #print("Model's Guess : ", label(query), "Correct answer : ", label(targets))
        if(label(query) == label(targets)):
            SLP_object.accuracy += 1
        SLP_object.train(inputs, targets)

    epochs_acc.append(SLP_object.accuracy/150)
    epochs_error.append(SLP_object.errors/150)

epochs_error = np.asfarray(epochs_error)
x = []
for i in range(0,epochs):
    x.append(i + 1)

plt.xlim(0,epochs)
plt.ylim(0,1)
plt.title("Error graph")
plt.xlabel('Epoch(s)')
plt.ylabel('Error')
plt.plot(x, epochs_error[:,0])
plt.plot(x, epochs_error[:,1])
plt.legend(["Node 1", "Node 2"], loc = "lower right")
plt.show()
plt.xlim(0,epochs)
plt.ylim(0,1)
plt.title("Accuracy graph")
plt.xlabel('Epoch(s)')
plt.ylabel('Accuracy')
plt.plot(x, epochs_acc)
plt.legend(["Accuracy"], loc = "lower right")
plt.show()