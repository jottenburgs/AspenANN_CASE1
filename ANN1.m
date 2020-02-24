%% Init
clear;
close all;
clc;

%% Parameter setup
input_layer_size = 3;
hidden_layer_size = 10;
num_labels=1;


%% Part 1: Load data
csv = readmatrix('InputDataCasestudy1.csv');
inputFeatures = 3;
[x,y,length]=GetData(csv,inputFeatures);

%% Part 2: Loading parameters
Theta_1 = - 3+ 6* rand(10,4);
Theta_2 = - 3+ 6* rand(1,11);
% Theta 1 (hidden layersize, inputlayersize+1) (+1 door bias)
% Theta 2 (num_labels, , hiddenlayersize+1) (+1 door bias)
nn_param= [Theta_1(:);Theta_2(:)];

%% Part 3: Normalize input/output
% Normalize input
x =normalizeInput (x);
% Normaize output
y =normalizeOutput(y);

%% Part 4: Compute cost
% Cost function
lambda=0;
[J, grad ] = ANNCostfunction (nn_param, input_layer_size, hidden_layer_size, num_labels, lambda, x, y);

fprintf('\nCost: %f\n', J);
fprintf('Gradients:\n');
fprintf(' %f \n', grad);

