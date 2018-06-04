close all
clear all
doplot = false

% Load data
d = load('mnist.mat');
trainX = double(d.trainX());
trainY = double(d.trainY());

preClassifyX = trainX(50000:end,:);
preClassifyY = trainY(1:length(preClassifyX));
trainX = trainX(1:50000-1,:);
trainY = trainY(1:length(trainX));

%Split Training data into data for each class
trainX_split = {};
for i=0:9
    trainX_classI = trainX(trainY==i,:);
    trainX_split{i+1} = double(trainX_classI);
end

%Calculate V for each class
V_split = {};
for i = 1:10
    [U,S,V] = svd(trainX_split{i},'econ');
    V_split{i}=V;
end

% %Calculate Z for each class
% Z_split = {};
% for i = 1:10
%     V = V_split{i};
%     X = trainX_split{i};
%     Z_i = arrayfun(@(j) X(j,:)*V,1:length(X),'UniformOutput',false);
%     Z_split{i} = double(cell2mat(Z_i'));
% end

%Classifie training data
[m,avgError_array] = classifie(preClassifyX,V_split,5,trainY,doplot,ones(1,10));

%% Classify & test data
testX = d.testX();
testY = d.testY();

%Use different p
pArray = 35;
relError_array = [];
for p = pArray

    %Classify
    classifiedLabel_array = classifie(testX,V_split,p,testY,doplot,avgError_array);

    % Calculate Error
    totalCorrect = sum(testY(1:length(classifiedLabel_array)) == classifiedLabel_array);
    classifiedNumber = arrayfun(@(j) sum(classifiedLabel_array==j),0:9)
    relativeError = 1-totalCorrect/length(classifiedLabel_array);
    relError_array = [relError_array, relativeError];
end

relError_array
figure(1);
plot(pArray, relError_array);

function plotWrongClasification(testSampleIndex, X_real,X_classified,X_correct,realLabel, classifiedLabel)
            figure(testSampleIndex)
            ax1 = subplot(1,3,1);
            imagesc(ax1, reshape(X_real,28,28)');
            title("realLabel: " + realLabel)
            ax2 = subplot(1,3,2);
            imagesc(ax2, reshape(X_classified,28,28)');
            title("classifiedLabel: " + classifiedLabel);
            ax3 = subplot(1,3,3);
            imagesc(ax3, reshape(X_correct,28,28)');
end

function unus = unusuabilty(X,V_i,Z_i)
    z = X*V_i;
    diff_array = arrayfun(@(j) immse(z, Z_i(j,1:length(z))),1:length(Z_i),'UniformOutput',false);
    diff_array = cell2mat(diff_array);
    unus = 1/sum(1./diff_array);
end
    
function [classifiedLabel_array, avgError_array] = classifie(testX, V_split,p,testY,doplot,avgError)
    classifiedLabel_array = [];
    totalError_array = zeros(1,10);
    totalWrong = 0;
    for testSampleIndex = 1:length(testX)
        error_Xapprox =[];
        %unus_Xapprox = [];
        X_approx_array = [];
        for i=0:9
            V = V_split{i+1};
            V = V(:,1:p);
            X = testX(testSampleIndex,:);
            X = double(X);
            X_approx = X*V*V';
            error = immse(X_approx,X);
            %unus = unusuabilty(X,V,Z_split{i+1});
            error_Xapprox = [error_Xapprox, error];
            %unus_Xapprox = [unus_Xapprox, unus];
            X_approx_array = [X_approx_array; X_approx];
        end
        error_Xapprox = error_Xapprox./avgError;        
        %unus_Xapprox = unus_Xapprox/sum(unus_Xapprox);
        %error_Xapprox = error_Xapprox/sum(error_Xapprox);
        %loss_Xapprox = unus_Xapprox + 5*error_Xapprox;
        [minError, minErrorIndex] = min(error_Xapprox);
        classifiedLabel = minErrorIndex - 1;
        totalError_array(classifiedLabel+1) = totalError_array(classifiedLabel+1)+minError;
        realLabel = testY(testSampleIndex);
        if classifiedLabel ~= realLabel
            totalWrong = totalWrong + 1;
            [double(testSampleIndex), double(classifiedLabel), double(realLabel), double(totalWrong)];
            if doplot
                plotWrongClasification(testSampleIndex,X,X_approx_array(classifiedLabel+1,:),...
                    X_approx_array(realLabel+1,:),realLabel,classifiedLabel); 
            end
        end
        classifiedLabel_array = [classifiedLabel_array, classifiedLabel];
    end
    sumPerClass = arrayfun(@(j) sum(classifiedLabel_array==j),0:9);
    avgError_array = totalError_array./sumPerClass;
end
