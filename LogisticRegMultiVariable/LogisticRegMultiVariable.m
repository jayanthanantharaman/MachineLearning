clear ; close all; clc

% Training Samples
trainData = csvread('G:\Courses\Machine_Learning\Assignment_2\TrainData.csv');
trainData =  trainData(randperm(end),:);

x = trainData(:, [1, 2]);
y = trainData(:, 5);

%x = featureScale(x);
m = length(y);

x = [ones(m, 1) x];
theta = zeros(3, 1);
MaxIter = 100;
lambda = 1000;

[cost, grad] = LogisticCostWithReg(x, y,theta, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
options = optimset('GradObj', 'on', 'MaxIter', MaxIter);

%  fminunc to obtain the optimal theta
%  This function will return theta and the cost
[theta, cost] = fminunc(@(theta)(LogisticCostWithReg(x, y,theta, lambda)), theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%Test Samples
testData = csvread('G:\Courses\Machine_Learning\Assignment_2\TestData.csv');
testData =  testData(randperm(end),:);

test_x = testData(:, [1, 2]); 
test_y = testData(:, 5);
%test_x = featureScale(test_x);

figure;
plot3(test_x(:,1),test_x(:,2),test_y,'k+', 'LineWidth', 1, 'MarkerSize', 15);
%axis([-inf,1.05,-0.05,1.05]);
ylabel('Recurrence'); 
xlabel('Texture,Area');

m = length(test_y);
test_x = [ones(m, 1), test_x];

threshold = 0.5 ;
test_result = classifyData(test_x,theta,threshold);

hold on;
plot3(test_x(:,2),test_x(:,3), test_result,'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
legend('Actual', 'Predicted')
hold off; 

[c,cm,ind,per] = confusion(test_y,test_result);
fprintf('Fraction of samples misclassified');
disp(c);

%Confusion Matrix
[C,order] = confusionmat(test_y,test_result);
disp(C);
disp(order);

%Calculate Precision Recall F1 Score
[precision,recall,f_score] = EvalMetrics(test_y,test_result);

fprintf('precision');
disp(precision);
fprintf('recall');
disp(recall);
fprintf('f_measure');
disp(f_score);
