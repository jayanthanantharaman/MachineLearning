clear ; close all; clc

% Training Samples
trainData = csvread('G:\Courses\Machine_Learning\Assignment_2\TrainMulti.csv');
trainData =  trainData(randperm(end),:);

x = trainData(:, 1); 
y = trainData(:, 5);
x = featureScale(x);

m = length(y);
x = [ones(m, 1), x]; 
theta = zeros(2, 1);
MaxIter = 100;

[cost,gradient] = LogisticCost(x, y,theta,m);

fprintf('Cost at initial theta (zeros): %f\n', cost);
options = optimset('GradObj', 'on', 'MaxIter', MaxIter);

%  fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(theta)(LogisticCost(x, y,theta,m)), theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%Test Samples
testData = csvread('G:\Courses\Machine_Learning\Assignment_2\TestMulti.csv');
testData =  testData(randperm(end),:);

test_x = testData(:, 1); 
test_y = testData(:, 5);
test_x = featureScale(test_x);


figure;
plot(test_x,test_y,'k+', 'LineWidth', 1, 'MarkerSize', 15);
axis([-inf,1.05,-0.05,1.05]);
ylabel('Recurrence'); 
xlabel('Texture'); 

m = length(test_y);
test_x = [ones(m, 1), test_x];

threshold = 0.2 ;
test_result = classifyData(test_x,theta,threshold);

hold on;
plot(test_x, test_result,'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
legend('Actual', 'Predicted')
hold off; 

[c,cm,ind,per] = confusion(test_y,test_result);
fprintf('Fraction of samples misclassified');
disp(c);

%Confusion Matrix
[C,order] = confusionmat(test_y,test_result);
disp(C);
disp(order);

[precision,recall,f_score] = EvalMetrics(test_y,test_result);

fprintf('precision');
disp(precision);
fprintf('recall');
disp(recall);
fprintf('f_measure');
disp(f_score);





    