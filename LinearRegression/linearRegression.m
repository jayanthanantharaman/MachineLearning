clear ; close all; clc

fprintf('Plotting Data ...\n')
data = csvread('G:\Courses\Machine_Learning\Assignment_1\BreastCancerDataTest.csv',140,0);

x = data(:, 1); 
y = data(:, 3);
m = length(y); 
x = featureScale(x);

figure;
plot(x, y, 'o'); 
ylabel('Compactness'); 
xlabel('Perimeter'); 

x = [ones(m, 1), x(:,1)]; 
%disp(x);

theta = zeros(2, 1); 

numberOfIterations = 250;
alpha = 0.06; 

% compute and display initial cost
cost(x, y, theta)

% run gradient descent
[theta,j] = gd(x, y, theta, alpha, numberOfIterations);

% compute and display final cost
cost(x, y, theta)

hold on;
plot(x, theta(1)*x+theta(2), '-');
legend('Training data', 'Linear regression')
hold off 

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

figure;
plot(1:numel(j), j, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');



% % Grid over which we will calculate J
% theta0_vals = linspace(-10, 10, 100);
% theta1_vals = linspace(-1, 4, 100);
% 
% % initialize J_vals to a matrix of 0's
% J_vals = zeros(length(theta0_vals), length(theta1_vals));
% 
% % Fill out J_vals
% for i = 1:length(theta0_vals)
%     for j = 1:length(theta1_vals)
%           t = [theta0_vals(i); theta1_vals(j)];    
%           J_vals(i,j) = cost(x, y, t);
%     end
% end
% 
% % Because of the way meshgrids work in the surf command, we need to 
% % transpose J_vals before calling surf, or else the axes will be flipped
% J_vals = J_vals';
% % Surface plot
% figure;
% surf(theta0_vals, theta1_vals, J_vals)
% xlabel('\theta_0'); ylabel('\theta_1');
% 
