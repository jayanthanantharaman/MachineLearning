function j = cost(x,y,theta)

m = length(y);

hypothesis= x * theta;
errorsSquared = (hypothesis - y).^2;

j= 1/(2*m) * sum(errorsSquared); 

end

