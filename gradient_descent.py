#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:12:39 2019

@author: musahni
"""

"""
Check neural network prediction with ground truth.
"""

knob_weight = 0.5
input = 0.5
goal = 0.8

prediction = knob_weight * input
# squared error
error = (goal - prediction)**2
print(error)

"""
HOT and COLD method
"""

# Let's make a prediction
weight = 0.1
lr = 0.01

def neural_network(input, weight):
    prediction = input*weight
    return prediction

number_of_toes = [8.5]
win_or_loss_binary = [1]

input = number_of_toes[0]
true = win_or_loss_binary[0]

prediction = neural_network(input, weight)
error = (prediction-true)**2
print(error)


# Let's make a prediction with a higher weight
prediction_high = neural_network(input, weight+lr)
error_high = (prediction_high - true)**2
print(error_high)


# Let's make a prediction with a lower weight and compare
prediction_low = neural_network(input, weight-lr)
error_low = (prediction_low - true)**2
print(error_low)

"""
Let's formulate a process around this:
iterate through increasing and decreasing the weight
till we can minimize the error between the prediction and
the ground truth
"""

weight = 0.5
input = 0.5
ground_truth = 0.8

step = 0.001

def squared_difference(prediction, truth):
    return (prediction-truth)**2

for i in range(1101):
    prediction = neural_network(input, weight)
    error = squared_difference(prediction, ground_truth)
    print(f"Iteration: {i} Error: {error} Prediction: {prediction}")
    
    high_prediction = neural_network(input, weight+step)
    high_error = squared_difference(high_prediction, ground_truth)
    
    low_prediction = neural_network(input, weight-step)
    low_error = squared_difference(low_prediction, ground_truth)
    
    if low_error < high_error:
        weight -= step
    else:
        weight += step
        

"""
Getting direction and error in the same line
"""

weight = 0.5
input = 0.5
ground_truth = 0.8

for i in range(20):
    prediction = neural_network(input, weight)
    error = (prediction-ground_truth)**2
    direction_and_amount = (prediction-ground_truth)*input
    weight -= direction_and_amount
    
    print(f"Iteration: {i} Error: {error} Prediction: {prediction}")
    
    
"""
Now we want to find the change in error with the change in weights.
For that we will use the chain rule.
"""

X = 0.5
W = 0.5
GT = 0.8

P = neural_network(X, W)
E = squared_difference(P, GT)

# Taking the derivative of the cost w.r.t. to the prediction
dE_dP = 2*(P-GT)

# Derivative of the prediction w.r.t. to the weight
dP_dW = input

# Combining the two to get dE/dW
dE_dW = dE_dP * dP_dW

# Let's update the weight now
alpha = 0.01
W -= dE_dW * alpha

W, GT, X = (0.0, 0.8, 0.5)

def weight_derivative(P, GT, X):
    dE_dP = (P-GT)
    dP_dW = X
    dE_dW = dE_dP*dP_dW
    return dE_dW
    
for i in range(50):
    P = neural_network(X, W)
    E = squared_difference(P, GT)
    W -= weight_derivative(P, GT, X)
    print(f"Iteration: {i} Error: {E} Prediction: {P}")

"""
Breaking the prediction
"""
W, GT, X = (0.0, 0.8, 2)

for i in range(50):
    P = neural_network(X, W)
    E = squared_difference(P, GT)
    W -= weight_derivative(P, GT, X)
    print(f"Iteration: {i} Error: {E} Prediction: {P}")

"""
Regularization
"""
alpha = 0.1
for i in range(50):
    P = neural_network(X, W)
    E = squared_difference(P, GT)
    W -= weight_derivative(P, GT, X)*alpha
    print(f"Iteration: {i} Error: {E} Prediction: {P}")
