function X_norm = featureScale(x)
X_norm = x;

xMax = max(x);

noOfIterations = size(X_norm,1);

for i= 1:noOfIterations 
X_norm(i,:) = X_norm(i,:)./ xMax ;
end

end
