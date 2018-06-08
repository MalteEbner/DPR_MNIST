close all
doplot = false

% Load data
d = load('../mnist.mat');

trainX = double(d.trainX());
trainX = reshape(trainX',28,28,1,60000);
trainY = categorical(d.trainY())';

testX = double(d.testX());
testX = reshape(testX',28,28,1,10000);
testY = categorical(d.testY())';

%Augment Data
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3])

imageSize = [28 28 1];
augimds = augmentedImageDatastore(imageSize,trainX,trainY,'DataAugmentation',imageAugmenter);


% Define network
layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(4,64,'Padding',1)
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,64,'Padding',1)
    dropoutLayer(0.6)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
     
    fullyConnectedLayer(1024)
    dropoutLayer(0.6)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    
    softmaxLayer
    classificationLayer];

%Define training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',500, ...
    'InitialLearnRate',0.03,'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod', 1,'LearnRateDropFactor', 0.8,...
    'L2Regularization',0.0005,...    
    'Verbose',true, ...
    'ValidationData', {testX,testY},...
    'ValidationPatience', 50, ...
    'Plots','training-progress');

%Train network
net = trainNetwork(augimds,layers,options);

%Classify and compute accuracy
YPred = classify(net,testX);

errorRate = 1- sum(YPred == testY)/numel(testY)

for i = 1:numel(testY)
    if YPred(i) ~= testY(i)
        if doplot
            plotWrongClassification(i,testX(:,:,1,i),testY(i),YPred(i));
        end
    end
end
            
            
function plotWrongClassification(testSampleIndex, X,realLabel, classifiedLabel)
    figure(testSampleIndex)
    imagesc(X');
    title("realLabel: " + (int32(realLabel)-1) + ", classifiedLabel: " + (int32(classifiedLabel)-1));
end