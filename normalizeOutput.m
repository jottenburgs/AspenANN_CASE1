function [ynorm] = normalizeOutput(y)
ymax = max(y);
ymin = min(y);
noutput=size(y,2);
ynorm=zeros(size(y));
for n=1:noutput
y(:,n) = (y(:,n)-ymin(1,n))/(ymax(1,n)-ymin(1,n));
ynorm(:,n) = y(:,n);
end
