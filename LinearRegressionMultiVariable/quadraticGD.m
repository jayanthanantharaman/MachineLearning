function[theta,j_old]= quadraticGD(x,y,theta,alpha,iterations)
m = length(y);
j_old = zeros(iterations,1);

for i=1:iterations
    
    hypothesis= x * theta;
    errors=hypothesis - y;
    
    decrement = (alpha * (1/m) * errors' *x);
    
    theta=theta - decrement';
    
    j_old(i)=quadraticCost(x,y,theta);
    
end

end
    
    