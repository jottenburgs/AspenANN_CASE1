function [xnorm] = normalizeInput(x)
xmax = max(x);
xmin = min(x);
ninput=size(x,2);
xnorm=zeros(size(x));

for n=1:ninput 
x(:,n) = (x(:,n)-xmin(1,n))/(xmax(1,n)-xmin(1,n));
xnorm(:,n) = x(:,n);
end
end

