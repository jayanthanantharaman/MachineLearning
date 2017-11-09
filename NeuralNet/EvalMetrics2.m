function EvalMetrics2(classes,expected,predicted)

[~, ex] = ismember(expected, classes,'rows') ;
[~, ob] = ismember(predicted, classes,'rows');

[C,order] = confusionmat(ex,ob);
disp(C);
disp(order);

numOfClasses = size(C,1);

[TP,TN,FP,FN,recall,precision,f_score] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
   TP(class) = C(class,class);
   tempMat = C;
   tempMat(:,class) = []; 
   tempMat(class,:) = []; 
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(C(:,class))-TP(class);
   FN(class) = sum(C(class,:))-TP(class);
end

for class = 1:numOfClasses
    recall(class) = TP(class) / (TP(class) + FN(class));
    precision(class) = TP(class) / (TP(class) + FP(class));
    f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
end

for class = 1:numOfClasses
fprintf('precision: %f %f\n',class, precision(class));
fprintf('recall: %f %f\n', class,recall(class));
fprintf('Fscore: %f %f\n', class,f_score(class));
end

end

