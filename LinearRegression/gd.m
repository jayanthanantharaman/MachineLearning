function[theta,j_old]= gd(x,y,theta,alpha,iterations)
m = length(y);
j_old = zeros(iterations,1);

for i=1:iterations
    hypothesis=x * theta;
    errors=hypothesis - y;
    
    x_col1= x(:,1);
    x_col2= x(:,2);
    
    theta(1, 1) = theta(1, 1) - (alpha * (1/m) * errors' * x_col1);
    theta(2, 1) = theta(2, 1) - (alpha * (1/m) * errors' * x_col2);
    
    j_old(i)=cost(x,y,theta);
    
%     fprintf('j(theta) iteration: %f',i);
%     fprintf(' %f ', j_old(i));
%     fprintf( '\n');
    
end

end
    
    