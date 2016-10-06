function res = classifyData(x,theta,threshold)

res = double(sigmoid( x * theta ) > threshold);

end