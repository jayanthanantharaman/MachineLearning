clear ; close all; clc

data = csvread('G:\Courses\Machine_Learning\Assignment_1\BreastCancerDataTest.csv',140,0);

X = data(:, 1); 
y = data(:, 3);
m = length(y);
X = featureScale(X);

figure; 
plot(X, y, 'o');      
ylabel('Compactness');           
xlabel('Perimeter'); 

x_quad = [ones(m, 1), X(:,1), X(:,1).^2];
disp(x_quad);

theta = zeros(3, 1); 
numberOfIterations = 250;
alpha = 0.06; 


quadraticCost(x_quad, y, theta)

[theta,j] = quadraticGD(x_quad, y, theta, alpha, numberOfIterations);

quadraticCost(x_quad, y, theta)

hold on;
plot(X, (theta(1)*(X.^2) + theta(2)*X + theta(3)), '-');
legend('Training data', 'Linear regression')
hold off 


fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

figure;
plot(1:numel(j), j, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


