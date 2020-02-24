function [sig] = sigmoid(a)
% Computes sigmoid of a
sig= 1.0 ./ (1.0 + exp(-a));
end

