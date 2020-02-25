%% Init
clear;
close all;
clc;

%% Parameter setup
input_layer_size = 3;
hidden_layer_size = 10;
num_labels=8;


%% Part 1: Load data
csv = readmatrix('InputDataCasestudy1.csv');
inputFeatures = 3;
[x,y,length]=GetData(csv,inputFeatures);

%% Part 2: Loading parameters
Theta_1 = - 3+ 6* rand(hidden_layer_size,input_layer_size+1);
Theta_2 = - 3+ 6* rand(num_labels,hidden_layer_size+1);
% Theta 1 (hidden layersize, inputlayersize+1) (+1 door bias)
% Theta 2 (num_labels, , hiddenlayersize+1) (+1 door bias)
nn_param= [Theta_1(:);Theta_2(:)];

%% Part 3: Normalize input/output
% Normalize input
x =normalizeInput (x);
% Normaize output
y =normalizeOutput(y);

%% Part 4: Compute cost + regularization
% Cost function
lambda=1;
[J, grad ] = ANNCostfunction (nn_param, input_layer_size, hidden_layer_size, num_labels, lambda, x, y);

fprintf('Cost:');
J
fprintf('Gradients:\n');
grad

%% Part 5: Initializing parameters

initial_Theta1 = - 3+ 6* rand(hidden_layer_size,input_layer_size+1);
initial_Theta2 = - 3+ 6* rand(num_labels,hidden_layer_size+1);
%randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Part 6: Training NN

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 1000);
costFunction = @ (p) ANNCostfunction(p, input_layer_size, hidden_layer_size, num_labels, lambda, x, y);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));

%% Part 7: Predict output
prediction = predict(Theta1,Theta2, x)
fprintf('Prediciton accuracy') ;
mean(double(prediction == y))*100





