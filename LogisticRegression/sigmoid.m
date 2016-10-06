function sig = sigmoid(z)

e=exp(1);
sig = 1./(1 + e.^(-z));

end