function [x,y,l] = GetData(csv, inputfeatures)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
sizeCSV= size(csv)
n= sizeCSV(:,2);
% n = number of columns
m = inputfeatures;
% m = total number of input features
x = csv (:,1:m);
y = csv (:,(m+1):n);
l = length(csv);
% l = number of input lines
end

