function [J grad] = ANNCostfunction(nn_param, input_layer_size, hidden_layer_size, num_labels, lambda, x, y)
% Parameters
inputnum = size(x,2);
outputnum = size (y,2);
m = size(x,1);


% Reshape theta
Theta1 = reshape(nn_param(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_param((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
% Init variables
J=0;

theta1_grad=zeros(size(Theta1));
theta2_grad=zeros(size(Theta2));

%% Part 1: Actual cost function
% Last hidden layer passes linear activation function
eye_matrix = eye(num_labels);
y_matrix =y;
%y_matrix = eye_matrix(y,:);

x = [ones(m,1) x];
a1 = x;
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = z3;

    temp1 = Theta1(:,2:size(Theta1,2));
    temp2 = Theta2(:,2:size(Theta2,2));
    reg = (lambda/(2*m))* (sum(sum(temp1.^2))+sum(sum(temp2.^2)));
    %placeholder: find a way to calculated error. For now total cross entropy E is
    %used

    J = J + (1/m)*sum(sum(((-y_matrix).*log(a3))-((1-y_matrix).*log(1-a3))))+reg;

%% Part 2:  Calculating  gradient

for i=1:m
   a1 = x(i,:);
   z2 = Theta1*a1';
   a2 = sigmoid(z2);
   a2 = [1; a2];
   z3 = Theta2*a2;
   a3=z3;
   
        d3 = a3 - y(i,:)';
        z2 = [1;z2];
        d2 = (Theta2'*d3).*sigmoid(z2);
        d2 = d2(2:end);
        
        theta2_grad = theta2_grad + d3*a2';
        theta1_grad = theta1_grad + d2*a1;
            
        
    
end

theta1_grad = (1/m)*theta1_grad;
theta2_grad = (1/m)*theta2_grad;

%% Part 3: Regularization

theta1_grad(:,2:end)= theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
theta2_grad(:,2:end)= theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

%% Part 4: Returning cost function
grad = [theta1_grad(:) ; theta2_grad(:)];

end

