clear ; close all; clc

fprintf('Loading data ...\n');

data = csvread('G:\Courses\Machine_Learning\Assignment_1\BreastCancerDataTest.csv',40,0);
x = data(:, 1:2);
%disp(x);
y = data(:, 3);
m = length(y);
x = featureScale(x);
%disp(x);

figure;
plot(x, y, 'o'); 
ylabel('Compactness'); 
xlabel('Perimeter,Area');

x = [ones(m, 1) x];
theta = zeros(3, 1);

quadraticCost(x, y, theta)
numberOfIterations = 500;
alpha = 0.05; 

[theta, j] = quadraticGD(x, y, theta, alpha, numberOfIterations);

% compute and display final cost
quadraticCost(x, y, theta)

hold on;
plot(x(:,2), (x * theta), '-');
legend('Perimeter','Area', 'Linear regression')
hold off 

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

figure;
plot(1:numel(j), j, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');