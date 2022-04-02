from distutils.log import error
from pickle import FALSE
import pandas as pd
import numpy as np
import math as mth
import random
import matplotlib.pyplot as plt

data_set = pd.read_excel("Ouse93-96 - Student.xlsx", header=1,usecols='A:J')
training_data = data_set[data_set["Date"].dt.strftime('%d%m%Y').str.contains("1993|1994")] #train on first 2 years
validation_data = data_set[data_set["Date"].dt.strftime('%d%m%Y').str.contains("1995")].reset_index(drop=True)
test_data =data_set[data_set["Date"].dt.strftime('%d%m%Y').str.contains("1996")].reset_index(drop=True)

#training take the first 2 years
#validation the next year
#test the last year

global learning_rate
learning_rate = 0.1 #small step size
global Error
Error = 0
global row
row = 0
global correct_output
correct_output = 0.00
max_Skelton = training_data["Skelton"].max()
min_Skelton = training_data["Skelton"].min()

max_Skelton_prev = training_data["Skelton - Previous Day"].max()
min_Skelton_prev = training_data["Skelton - Previous Day"].min()

max_Westwick = training_data["Westwick"].max()
min_Westwick = training_data["Westwick"].min()

max_Skip_Bridge = training_data["Skip Bridge"].max()
min_Skip_Bridge = training_data["Skip Bridge"].min()

max_Crakehill = training_data["Crakehill"].max()
min_Crakehill = training_data["Crakehill"].min()

#function for the sygmoid function
def sigmoid(x):
    return 1 / (1 + mth.exp(-x))
#function to standardise data to [0.1,0.9]
def standardise(R,max,min):
    S = 0.8*((R-min)/(max-min)) + 0.1
    return S

def de_standardise(S,max,min):
    R = (((S-0.1)/0.8)*(max-min)) + min
    return R
def RMSE(value,size):
    out = (value/size)**0.5
    return out
def assign_inputs(index,data_set):
        correct_output = standardise(data_set.loc[index,"Skelton"],max_Skelton,min_Skelton)
        input_layer[0].input = standardise(data_set.loc[index,"Crakehill"],max_Crakehill,min_Crakehill)
        input_layer[1].input = standardise(data_set.loc[index,"Skip Bridge"],max_Skip_Bridge,min_Skip_Bridge)
        input_layer[2].input = standardise(data_set.loc[index,"Westwick"],max_Westwick,min_Westwick)
        input_layer[3].input = standardise(data_set.loc[index,"Skelton - Previous Day"],max_Skelton_prev,min_Skelton_prev)
        return correct_output
def forward_pass(data):
    err = 0.0
    for i in range(len(data.index)):
        correct_output = assign_inputs(i,data)
        node_cnt = 0
        for node in input_layer:
            node.update_output()
            for y in range(len(hidden_layer)):
                hidden_layer[y].inputs[node_cnt] = node.outputs[y]
            node_cnt += 1
        for x in range (len(hidden_layer)):
            output_node.inputs[x] = hidden_layer[x].node_output()
        err += (de_standardise(output_node.node_output(),max_Skelton,min_Skelton) - de_standardise(correct_output,max_Skelton,min_Skelton))**2
    return(RMSE(err,len(data.index)))
def update_nodes():
    node_cnt = 0
    for node in input_layer:
        node.update_output()
        for y in range(len(hidden_layer)):
            hidden_layer[y].inputs[node_cnt] = node.outputs[y]
        node_cnt += 1
    for x in range (len(hidden_layer)):
        output_node.inputs[x] = hidden_layer[x].node_output()
correct_output = standardise(data_set.loc[row,"Skelton"],max_Skelton,min_Skelton)

class Output_Node:
    def __init__(self,bias):
        self.bias = bias
        self.prev_bias = float
        self.inputs = [] #These will be values of w i,j * u i
        self.sig_prime = float #value of f'(Sj)
        self.U = float #value of Uj
        self.delta = float #DELTA j
    def node_output(self):
        weighted_sum = sum(self.inputs) + self.bias #Sj
        self.U = sigmoid(weighted_sum) 
        return self.U  
    def get_delta(self):
        return self.delta
    def update_weights(self): #update bias for output node
        self.prev_bias = self.bias
        self.sig_prime = self.U*(1-self.U) #f'(Sj)
        self.delta = (correct_output - self.U) * self.sig_prime # (DELTA j = W j,o * DELTA o * f'(Sj) )
        self.bias = self.bias + (learning_rate * (self.delta))

class Hidden_Node:
    def __init__(self, bias, weight):
        self.bias = bias
        self.weight = weight #The value for the weight that will be used with the output
        self.inputs = [] #These will be values of w i,j * u i
        self.sig_prime = float #value of f'(Sj)
        self.U = float #value of Uj
        self.delta = float #DELTA j
        self.output = float
        self.prev_weight = weight
        self.prev_bias = float
     
    def node_output(self):
        weighted_sum = sum(self.inputs) + self.bias #Sj
        self.U = sigmoid(weighted_sum) 
        self.output = self.U * self.weight
        return self.output
    
    def get_delta(self):
        return self.delta
    
    def update_weights(self, delta_out):
        self.prev_bias = self.bias
        self.prev_weight = self.weight
        self.sig_prime = self.U * (1-self.U) #f'(Sj)
        self.delta = self.weight * delta_out * self.sig_prime # (DELTA j = W j,o * DELTA o * f'(Sj) )
        self.weight += (learning_rate * (delta_out*self.U))
        self.weight += 0.9*(self.weight-self.prev_weight) #value for alpha is 0.9
        self.bias += (learning_rate * (self.delta))

class Input_Node:
    def __init__(self, input, number_of_hidden):
        self.input = input
        self.number_of_hidden = number_of_hidden #number of hidden nodes we are using
        self.weights = []
        self.prev_weights = []
        self.outputs = []
        self.sig_prime = float
        self.assign_weights()
        self.node_output()

    def node_output(self):
        for i in range(len(self.weights)):
            self.outputs.append(self.weights[i] * self.input) 
    
    def assign_weights(self):
        for i in range(self.number_of_hidden):
            self.weights.append(random.uniform(-2/input_nodes,2/input_nodes))
        self.prev_weights = self.weights.copy()
    
    def update_weights(self,delta_next,weight_to_update): 
        self.prev_weights[weight_to_update] = self.weights[weight_to_update]
        self.weights[weight_to_update] += learning_rate * (self.input*delta_next)
        self.weights[weight_to_update] += 0.9*(self.weights[weight_to_update]-self.prev_weights[weight_to_update])
        self.outputs[weight_to_update] = self.weights[weight_to_update] * self.input

    def update_output(self):
        for i in range(len(self.outputs)):
            self.outputs[i] = self.weights[i] * self.input

    def revert_weights(self):
        self.weights = self.prev_weights.copy()
        self.update_output()

#using 4 input nodes, 2 hidden nodes and one output with range for weights and biases being [-0.5,0.5]
#generate nodes and randomly assign weights and biases

hidden_nodes = 8 #number of hidden nodes
global input_nodes
input_nodes = 4
input_layer = []
input_layer.append(Input_Node(standardise(training_data.loc[0,"Crakehill"],max_Crakehill,min_Crakehill),hidden_nodes))
input_layer.append(Input_Node(standardise(training_data.loc[0,"Skip Bridge"],max_Skip_Bridge,min_Skip_Bridge),hidden_nodes))
input_layer.append(Input_Node(standardise(training_data.loc[0,"Westwick"],max_Westwick,min_Westwick),hidden_nodes))
input_layer.append(Input_Node(standardise(training_data.loc[0,"Skelton - Previous Day"],max_Skelton_prev,min_Skelton_prev),hidden_nodes))


hidden_layer = []
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))
hidden_layer.append(Hidden_Node(random.uniform(-2/input_nodes,2/input_nodes),random.uniform(-2/input_nodes,2/input_nodes)))

output_node = Output_Node(random.uniform(-2/input_nodes,2/input_nodes))

#provide hidden nodes with weights and inputs from input nodes
for node in input_layer:
    for i in range(len(hidden_layer)):
        hidden_layer[i].inputs.append(node.outputs[i])

for node in hidden_layer:
    output_node.inputs.append(node.node_output())

epochs = 0
epoch_array = []
error_array = []
RMSE_array = []
Error = 0.00
prev_validation_error = forward_pass(validation_data)
prev_training_error = forward_pass(training_data)
prev_training_error_increased = False
error_increased = False
size_data = len(training_data.index)

while not error_increased:
    Error += (de_standardise(output_node.node_output(),max_Skelton,min_Skelton) - de_standardise(correct_output,max_Skelton,min_Skelton))**2
    #backward pass
    output_node.update_weights()
    delta_out = output_node.get_delta()
    #update weights and biases for hidden nodes and input nodes
    cnt = 0
    for node in hidden_layer:
        node.update_weights(delta_out)
        for i in range(len(input_layer)):
            input_layer[i].update_weights(node.get_delta(),cnt)
        cnt += 1

    if row == size_data-1: #single epoch has been completed
        row=0
        epochs +=1
        #print(epochs)
        epoch_array.append(epochs)
        Error = RMSE(Error,size_data)
        RMSE_array.append(Error)
        Error = 0.00
        if epochs % 1000 == 0 and epochs!= 0:
            validation_error = forward_pass(validation_data)
            if prev_validation_error < validation_error:
                error_increased = True
            else:
                prev_validation_error = validation_error

            #Bold driver
            training_error = forward_pass(training_data)
            #print(training_error)
            if prev_training_error * 1.04 < training_error: #training error increased by over 4%
                prev_training_error_increased = True
                while prev_training_error_increased:
                    #revert weight changes and change the learning rate
                    output_node.bias = output_node.prev_bias

                    for node in hidden_layer:
                        node.bias = node.prev_bias
                        node.weight = node.prev_weight

                    for i in range(len(input_layer)):
                        input_layer[i].revert_weights()

                    update_nodes()
                    learning_rate = learning_rate*0.7 #reduce the learning rate
                    training_error = forward_pass(training_data)

                    if prev_training_error * 1.04 < training_error:
                        if learning_rate<=0.01:#restrict learning parameter minimum to 0.01
                            learning_rate = 0.01
                        prev_training_error_increased = False
                    else:
                        prev_training_error = training_error
                        
            else: #learning rate may be too small so is increased
                learning_rate = learning_rate * 1.05
                if learning_rate > 0.5:
                    learning_rate = 0.5
            prev_training_error = training_error
            print(learning_rate)
    row += 1
    correct_output = assign_inputs(row,training_data)
    #change inputs for hidden nodes and output node
    update_nodes()

#test data
f = open("8 hidden nodes.txt","a")
test_error = 0.0
test_predictors = []
test_predictands = []
mse_error = 0.00
for i in range(len(test_data.index)):
    correct_output = assign_inputs(i,test_data)
    node_cnt = 0
    for node in input_layer:
        node.update_output()
        for y in range(len(hidden_layer)):
            hidden_layer[y].inputs[node_cnt] = node.outputs[y]
        node_cnt += 1
    for x in range (len(hidden_layer)):
        output_node.inputs[x] = hidden_layer[x].node_output()
    test_predictors.append(de_standardise(output_node.node_output(),max_Skelton,min_Skelton))
    test_predictands.append(de_standardise(correct_output,max_Skelton,min_Skelton))
    mse_error += ((de_standardise(output_node.node_output(),max_Skelton,min_Skelton) - de_standardise(correct_output,max_Skelton,min_Skelton))/de_standardise(correct_output,max_Skelton,min_Skelton))**2
    test_error += (de_standardise(output_node.node_output(),max_Skelton,min_Skelton) - de_standardise(correct_output,max_Skelton,min_Skelton))**2

print("RMSE: %f" % (RMSE(test_error,len(test_data))))
print("MSE: %f" % (mse_error/len(test_data)))
f.close()
test_results = pd.DataFrame({
    'Modelled': test_predictors,
    'Observed': test_predictands
})
file_name = 'Scatter.xlsx'
test_results.to_excel(file_name)
    
plt.title("ERROR")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.plot(epoch_array,RMSE_array)
plt.show()
    

