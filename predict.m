function prediction = predict(Theta1,Theta2,x)
% Function used to dermine prediction based on trained weight of NN
m =size(x,1);
num_labels=size(Theta2, 1);

prediction = zeros(size(x,1),num_labels);

x2 = [ones(m,1) x];
a1 =x2;
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = z3;

prediction = a3;


