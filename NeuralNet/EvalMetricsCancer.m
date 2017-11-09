function [precision,recall,f_score] = EvalMetricsCancer(ACTUAL,PREDICTED)

%Confusion Matrix
[C,order] = confusionmat(ACTUAL,PREDICTED);
disp(C);
disp(order);

idx = (ACTUAL()==1);

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));


tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;

precision = tp/(tp+fp);
recall = tp/p;
f_score = 2*((precision*recall)/(precision + recall));

fprintf('precision');
disp(precision);
fprintf('recall');
disp(recall);
fprintf('f_measure');
disp(f_score);




