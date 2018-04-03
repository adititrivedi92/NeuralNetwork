"""
Implementation of Resilient Backpropagation Neural Networks (Delta Rule)
"""
#from math import exp
import numpy as np
import math
from random import random
from csv import reader
from random import randrange
from collections import Counter

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
#    print (n_inputs, n_outputs, n_hidden)
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)], 'delta':0.8} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)],'delta':0.8} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Split a dataset into k folds
def cross_validation_split(dataset_test, no_folds):
    dataset_split = list()
    dataset_copy = list(dataset_test)
    fold_size = int(len(dataset_test) / no_folds)
    for i in range(no_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy
def accuracy_calculation(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def back_propagation_process(train, test, learning_rate, iteration, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, learning_rate, iteration, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


# Evaluate an algorithm
def evaluate_algorithm(dataset_train, dataset_test, algorithm, no_folds, *args):
    folds = cross_validation_split(dataset_test, no_folds)
    scores = list()
    for fold in folds:
        actual = [row[-1] for row in fold]
        predicted = algorithm(dataset_train, dataset_test, *args)
        resPred = Counter(predicted).most_common(1)[0][0]
        resAct = Counter(actual).most_common(1)[0][0]
        prediction = [resAct if x == resPred else resPred for x in predicted]
        accuracy = accuracy_calculation(actual, prediction)
        scores.append(accuracy)
    return scores

 
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i] * momentum_alpha
    return activation
 
# Transfer neuron activation: tanh function
#def transfer(activation):
#    return np.tanh(activation)

# Transfer neuron activation: logistic sigmoid function
def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Calculation of Backpropagate Error
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron_delta = errors[j] * transfer_derivative(neuron['output'])
            if neuron_delta > neuron['delta']:
                neuron['delta'] = neuron_delta
   

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]+momentum_alpha
            neuron['weights'][-1] += l_rate * neuron['delta']    
 
 
# Train a network for a fixed number of iteration
def train_network(network, train, learning_rate, iteration, n_outputs):
    for epoch in range(iteration):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [1 for i in range(n_outputs)]
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)
  
 

# Prediction Calculation
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

 
# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip()) 

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
 
 

# Input data set
filename = 'digit_train_4.csv'
dataset_train = load_csv(filename)

# Input data set
filename2 = 'digit_test_4.csv'
dataset_test = load_csv(filename)

# convert string attributes to integers
for i in range(len(dataset_train[0])):
    str_column_to_float(dataset_train, i)

# convert string attributes to integers
for i in range(len(dataset_test[0])):
    str_column_to_float(dataset_test, i)

# convert class column to integers
str_column_to_int(dataset_train, len(dataset_train[0])-1)

# convert class column to integers
str_column_to_int(dataset_test, len(dataset_test[0])-1)

X_train = list()
X_test = list()

X_train = dataset_train
X_test = dataset_test

#iteration = [100, 110, 120, 130]

no_of_folds = 3
learning_rate = [0.05, 0.1, 0.3, 0.5, 0.8] 
n_hidden = [4,16]#no of neurons in hiddent layer
momentum_alpha = 0.3 #[0.1, 0.3, 0.9]  # bias for adjusting weights 0.9 to .99
iteration = 10;

print('Weight Momentum : ', momentum_alpha)
for index1 in range(len(n_hidden)):
    print('no of neurons in hidden layer :', n_hidden[index1])
  
    for index in range(len(learning_rate)):
        classificationAccuracy = evaluate_algorithm(X_train, X_test, back_propagation_process, no_of_folds, learning_rate[index], iteration, n_hidden[index1])
        print('Learning Rate : ', learning_rate[index])
#       print('No of Hidden Layer : ', n_hidden[index])
#       print('Weight Momentum : ', momentum_alpha)
#       print('No of Iteration: ', iteration)   
        print('classification Accuracy : %s' % max(classificationAccuracy))