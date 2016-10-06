function [precision,recall,f_score] = EvalMetrics(ACTUAL,PREDICTED)

idx = (ACTUAL()==1);

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));


tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;

precision = tp/(tp+fp);
recall = tp/p;
f_score = 2*((precision*recall)/(precision + recall));


