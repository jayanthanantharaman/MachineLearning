function [cost, grad] = LogisticCostWithReg(x, y,theta, lambda)

noOfSamples = length(y);

gradient = zeros(size(theta)); 
hypothesis = sigmoid(x*theta);

costJ = (-1/noOfSamples) * sum( y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis) );

costReg = lambda/(2*noOfSamples) * sum( theta(2:end).^2 );

cost = costJ + costReg;


for i = 1:noOfSamples
	gradient = gradient + ( hypothesis(i) - y(i) ) * x(i, :)';
end

gradientReg = lambda/noOfSamples * [0; theta(2:end)]; 

grad = (1/noOfSamples) * gradient + gradientReg;

end