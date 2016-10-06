function[cost,gradient] = LogisticCost(x,y,theta,noOfSamples)

gradient= zeros(size(theta));

hypothesis = sigmoid(x*theta);

cost = (-1/noOfSamples)* sum(y.* log(hypothesis) + (1-y).* log(1-hypothesis));

for i=1:noOfSamples
    gradient=gradient+(hypothesis(i)-y(i)) * x(i,:)';
end

gradient = (1/noOfSamples)*gradient;

end