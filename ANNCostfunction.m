function [J grad] = ANNCostfunction(nn_param, input_layer_size, hidden_layer_size, num_labels, lambda, x, y)
% Parameters
inputnum = size(x,2);
outputnum = size (y,2)
m = length(x);


% Reshape theta
Theta1 = reshape(nn_param(1:hidden_layer_size*(input_layer_size +1)), hidden_layer_size, (input_layer_size +1));
Theta2 = reshape(nn_param((1+(hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size +1));
% Init variables
J=zeros(outputnum,1);

for i=1:outputnum
    theta1_grad(:,:,i)=zeros(size(Theta1));
    theta2_grad(:,:,i)=zeros(size(Theta2));
end

%% Part 1: Actual cost function

x = [ones(m,1) x];
a1 = x;
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = z3;

% Last hidden layer passes linear activation function
for p=1:outputnum
    temp1 = Theta1(:,2:size(Theta1,2));
    temp2 = Theta2(:,2:size(Theta2,2));
    reg = (lambda/(2*m))* (sum(sum(temp1.^2))+sum(sum(temp2.^2)));
    J(p,1) = J(p,1) + (1/m)*sum(sum(((-y(:,p))'.*log(a3))-((1-y(:,p))'*log(1-a3)))) + reg;
    
    
end


%% Part 2:  Calculating  gradient
%d = input_layer_size+1;
%f = hidden_layer_size+1;
%g = num_labels;

for i=1:m
    a1 = x(i,:);
    z2 = Theta1*a1';
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2*a2;
    a3 = z3;
    z2= [1;z2];
    for j=1:outputnum
        
        
        d3 = a3 - y(i,j);
        fprintf('d3');
        size(d3)
        fprintf('thetaac');
        size(Theta2')
        fprintf('voorsig');
        sigmoidGradient(z2)
        fprintf('voorsig');
        sigmoidGradient(z2)
        d2 = (Theta2'*d3).*sigmoidGradient(z2);
        d2 = d2(2:end);
            theta1_grad = theta1_grad + d2*a1;
            theta2_grad = theta2_grad + d3*a2;
        
    end
end

theta1_grad = (1/m)*theta1_grad;
theta2_grad = (1/m)*theta2_grad;

%% Part 3: Regularization
size(theta1_grad(:,2:end))
size((lambda/m)*Theta1(:,2:end))
size(theta2_grad(:,2:end))
size((lambda/m)*Theta2(:,2:end))
theta1_grad(:,2:end)= theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
theta2_grad(:,2:end)= theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

%% Part 4: Returning cost function
grad = [theta1_grad(:) ; theta2_grad(:)];

end

