 clear ; close all; clc

iterations = 1000;
errorThreshhold = 0.1;
learningRate = 0.5;
hiddenNeurons = [13 13 2];
hidden_layer_size = 2;
threshold = 0.5;
classes = [1 0 0;0 1 0;0 0 1];

trainData = load('G:\Courses\Machine_Learning\Assignment_3\wineTrain.data');
% trainData = csvread('G:\Courses\Machine_Learning\Assignment_3\BreastCancerDataTrain.csv');

trainInp = trainData(:, 2:end);
% trainInp = trainData(:, 2:6);
trainInp = featureScale(trainInp);
trainOut = trainData(:,1);

trainRealOut = zeros(length(trainOut),size(trainOut, 2));
for i=1:size(trainOut)
    trainRealOut(i,trainOut(i)) = 1;
end

feature_Count = size(trainInp, 2);
% output_layer_cnt = size(trainOut, 2);
output_layer_cnt = size(trainRealOut, 2);
m = size(trainInp, 1);
layerOfNeurons = [hiddenNeurons, output_layer_cnt];
layerCount = size(layerOfNeurons, 2);

e = 0;
b = 0;

weightCell = cell(1, layerCount);
for i = 1:layerCount
    if i == 1
        weightCell{1} = unifrnd(b, e, feature_Count,layerOfNeurons(1));
    else
        weightCell{i} = unifrnd(b, e, layerOfNeurons(i-1),layerOfNeurons(i));
    end
end


biasCell = cell(1, layerCount);
for i = 1:layerCount
    biasCell{i} = unifrnd(b, e, 1, layerOfNeurons(i));
end

%---Train
for iter = 1:iterations
    for i = 1:m
        sampleIn = trainInp(i, :);
        sampleTarget = trainRealOut(i, :);
%         sampleTarget = trainOut(i, :);
        [realOutput, layerOutputCells] = FwdPropagate(sampleIn, layerOfNeurons, weightCell, biasCell);
        [weightCell, biasCell] = BackPropagate(learningRate, sampleIn,sampleTarget, layerOfNeurons,weightCell, biasCell, layerOutputCells);
    end
    
    error = zeros(m, output_layer_cnt);
    p=zeros(m, output_layer_cnt);
    err=zeros(m, output_layer_cnt);
  
    for t = 1:m
        [predict, layeroutput] = FwdPropagate(trainInp(t, :), layerOfNeurons, weightCell, biasCell);
        p(t,:) = predict;
        error(t, : ) = predict - trainRealOut(t, :);
%          error(t, : ) = predict - trainOut(t, :);
    end
    err(iter,:) = (sum(error.^2)/m).^0.5;
    if err(iter) < errorThreshhold
        break;
    end
end
 fprintf('Ended with %d iterations.\n', iter);
 
%Test Data
testData = load('G:\Courses\Machine_Learning\Assignment_3\wineTest.data');
% testData = csvread('G:\Courses\Machine_Learning\Assignment_3\BreastCancerDataTest.csv');
% test_x = testData(:, 2:6);
test_x = testData(:, 2:end);
test_y = testData(:,1);

test_x = featureScale(test_x);

target_y = zeros(length(test_y),size(test_y, 2));
for i=1:size(test_y)
    target_y(i,test_y(i)) = 1;
end

testSamples = size(test_x, 1);
error = zeros(testSamples, output_layer_cnt);
test_result = zeros(testSamples, output_layer_cnt);
for t = 1:testSamples
    [predict, layeroutput] = FwdPropagate(test_x(t, :), layerOfNeurons, weightCell, biasCell);
    test_result(t,:) = double(predict > threshold);
    error(t, : ) = predict - target_y(t, :);
%     error(t, : ) = predict - test_y(t, :);
end

% figure(1);
% % p1 = plot3(test_x(:,1),test_x(:,2),target_y,'k+', 'LineWidth', 1, 'MarkerSize', 15,'DisplayName','Actual');
% plot3(test_x(:,1),test_x(:,2),test_y,'k+', 'LineWidth', 1, 'MarkerSize', 15);
% 
% zlabel('Wine Classes'); 
% xlabel('Alcohol');
% ylabel('Malic acid');
% hold on;
% p3 = plot3(test_x(:,1),test_x(:,2), test_result,'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 5,'DisplayName','Predicted');
% hold off;

[c,cm,ind,per] = confusion(target_y,test_result);
% [c,cm,ind,per] = confusion(test_y,test_result);

fprintf('Fraction of samples misclassified');
disp(c);


 EvalMetrics2(classes,target_y,test_result);
% EvalMetricsCancer(test_y,test_result);



