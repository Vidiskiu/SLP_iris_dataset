import numpy as np
import matplotlib.pyplot as plt

def round(x):
    for i in range(len(x)):
        if(x[i]>=0.5):
            x[i] = 1
        else:
            x[i] = 0
    return x

def encode(target):
    if target == 'setosa':
        return np.asfarray([0.0,0.0])
    elif target == 'versicolor':
        return np.asfarray([0.0,1.0])
    elif target == 'virginica':
        return np.asfarray([1.0,1.0])
    else:
        print("Target not Found")

def listDifference(list_A, list_B):
    temp_list = [item for item in list_A if item not in list_B]
    final_list = []
    for segment in temp_list:
        final_list.extend(segment)
    return final_list

class MLP:
    def __init__(self, input, hidden, output, learning_rate):
        # Number of nodes
        self.input = input
        self.hidden = hidden
        self.output = output

        # Learning rate
        self.lr = learning_rate
        
        # List of errors
        self.epoch_error = np.zeros((self.output,1))
        self.epoch_errors = []
        self.epoch_errors_val = []

        # List of accuracy
        self.epoch_accuracy = 0
        self.epoch_accuracies = []
        self.epoch_accuracies_val = []

        # Random initial weights
        self.wih = np.random.rand(self.hidden, self.input)
        self.wih_bias = np.random.rand(self.hidden, 1)

        self.who = np.random.rand(self.output, self.hidden)
        self.who_bias = np.random.rand(self.output, 1)
    
        # Defining sigmoid as activation function
        self.sigmoid = lambda x : (1 / (1 + np.exp(-1 * x)))

    def train(self, inputs, target):
        # inputs array transposed for calculation
        inputs = np.array(inputs,ndmin=2).T
        target = np.array(target,ndmin=2).T

        # Inputs calculation to the nodes and its outputs
        hidden_inputs = np.dot(self.wih, inputs)
        # Add bias
        hidden_inputs = hidden_inputs + self.wih_bias
        # Activation function
        hidden_outputs = self.sigmoid(hidden_inputs)

        # Inputs calculation to the output nodes and its outputs
        outputs_inputs = np.dot(self.who, hidden_outputs)
        # Add bias
        outputs_inputs = outputs_inputs + self.who_bias
        outputs_outputs = self.sigmoid(outputs_inputs)

        # calculating the errors of the outputs
        outputs_error = target - outputs_outputs
        self.epoch_error = self.epoch_error + (0.5* (outputs_error) **2)

        # calculating the errors of the hidden layer, back propagation of errors
        hidden_error = np.dot(self.who.T, outputs_error)

        # updating the weights
        self.who += self.lr * np.dot(((outputs_error) * (outputs_outputs) * (1 - outputs_outputs)), np.array(hidden_outputs).T)
        self.who_bias += self.lr * ((outputs_error) * (outputs_outputs) * (1 - outputs_outputs))

        self.wih += self.lr * np.dot((hidden_error * (hidden_outputs) * (1 - hidden_outputs)), np.array(inputs).T)
        self.wih_bias += self.lr * ((hidden_error) * (hidden_outputs) * (1 - hidden_outputs))

        pred = round(outputs_outputs)

        if(pred[0] == target[0] and pred[1] == target[1]):
            self.epoch_accuracy = self.epoch_accuracy + 1

    # Saving average error from each epochs
    def save_error(self):
        temp_error = 0
        
        for i in range(self.output):
            temp_error = temp_error + self.epoch_error[i]/120

        self.epoch_errors.append(temp_error)
        self.clearEpochError()

        # Saving average error from each epochs
    def save_errorVal(self):
        temp_error = 0
        
        for i in range(self.output):
            temp_error = temp_error + self.epoch_error[i]/30

        self.epoch_errors_val.append(temp_error)
        self.clearEpochError()

    # Saving average acc from each epochs
    def save_acc(self):
        self.epoch_accuracies.append(self.epoch_accuracy/120)
        self.clearEpochAcc()

    # Saving average acc from each epochs
    def save_accVal(self):
        self.epoch_accuracies_val.append(self.epoch_accuracy/30)
        self.clearEpochAcc()

    # Clear variable for next epoch
    def clearEpochError(self):
        self.epoch_error = np.zeros((self.output,1))

    # Clear variable for next epoch
    def clearEpochAcc(self):
        self.epoch_accuracy = 0
    
    # Get epoch errors
    def getEpochErrors(self):
        return self.epoch_errors

    # Get epoch errors for val
    def getEpochErrorsVal(self):
        return self.epoch_errors_val

    # Get epoch accuracy
    def getEpochAccuracies(self):
        return self.epoch_accuracies

    # Get epoch accuracy for val
    def getEpochAccuraciesVal(self):
        return self.epoch_accuracies_val

    def getStats(self):
        print("wih",self.wih,"\nb1",self.wih_bias,"\nwho",self.who,"\nb1",self.who_bias)
    def getStats2(self):
        print("max acc", max(self.epoch_accuracies), "\nmax val", max(self.epoch_accuracies_val), "\nAvg acc", sum(self.epoch_accuracies)/epochs, "\nAvg acc val", sum(self.epoch_accuracies_val)/epochs)
        print("min err", min(self.epoch_errors), "\nmin err v",min(self.epoch_errors_val),"\navg err",sum(self.epoch_errors)/epochs,"\navg err v",sum(self.epoch_errors_val)/epochs)

# ========================================================================================
# Main Body
# ========================================================================================

# MLP architecture and hyperparameter
input = 4
hidden = 4
output = 2
learning_rate = 0.05
epochs = 100

# Opening dataset
file = open('iris.csv','r')
dataset = file.readlines()
file.close()

# Removing first column
dataset = dataset[1:]

# MLP Object
MLP_object = MLP(input, hidden, output, learning_rate)

MLP_object.getStats()

kdataset = []
k = 5

for i in range(k):
    kdataset.append([])
       
counter = 0

# Distribute dataset
for record in dataset:
    kdataset[counter % k].append(record)
    counter += 1

# Split data set into 120:30 Training:Validation
dataset_val = []
dataset_val.append(kdataset[4])
dataset_train = listDifference(kdataset, dataset_val)
dataset_val = dataset_val[0]

for e in range(epochs):
    for record in dataset_train:
        # Fetch features
        values = record.split(",")
        inputs = np.asfarray(values[0:4])

        # Fetch target value and encode target
        target = encode(values[4])
        MLP_object.train(inputs, target)

    MLP_object.save_error()
    MLP_object.save_acc()

    for record in dataset_val:
        # Fetch features
        values = record.split(",")
        inputs = np.asfarray(values[0:4])

        # Fetch target value and encode target
        target = encode(values[4])
        MLP_object.train(inputs, target)

    MLP_object.save_errorVal()
    MLP_object.save_accVal()

# Draw graph

x = np.linspace(1,epochs,epochs)

errors = MLP_object.getEpochErrors()
errors_val = MLP_object.getEpochErrorsVal()
acc = MLP_object.getEpochAccuracies()
acc_val = MLP_object.getEpochAccuraciesVal()

MLP_object.getStats2()

plt.xlim(0,epochs)
plt.ylim(0,1)

plt.title("Error graph")
plt.xlabel('Epoch(s)')
plt.ylabel('Error')
plt.plot(x,errors)
plt.plot(x,errors_val)
plt.legend(["Training Error", "Validation Error"], loc = "lower right")
plt.show()

plt.title("Accuracy graph")
plt.xlabel('Epoch(s)')
plt.ylabel('Accuracy')
plt.plot(x,acc)
plt.plot(x,acc_val)
plt.legend(["Training Accuracy", "Validation Accuracy"], loc = "lower right")
plt.show()