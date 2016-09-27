clear ; close all; clc

fprintf('Loading data ...\n');

data = csvread('G:\Courses\Machine_Learning\Assignment_1\BreastCancerDataTest3.csv',140,0);
x = data(:, 1:3);
%disp(x);
y = data(:, 4);
m = length(y);
x = featureScale(x);
%disp(x);

figure;
plot(x, y, 'o'); 
ylabel('Compactness'); 
xlabel('Perimeter,Area');

x = [ones(m, 1) x];
theta = zeros(4, 1);

quadraticCost(x, y, theta)
numberOfIterations = 500;
alpha = 0.7;

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