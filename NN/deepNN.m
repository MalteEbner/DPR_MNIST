close all

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

    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%Define training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'InitialLearnRate',0.03,'L2Regularization',0.0005,'LearnRateSchedule','piecewise',...
    'Verbose',true, ...
    'ValidationData', {testX,testY},...
    'ValidationPatience', 10, ...
    'Plots','training-progress');

%Train network
%net = trainNetwork(augimds,layers,options);

%Classify and compute accuracy
YPred = classify(net,testX);

errorRate = 1- sum(YPred == testY)/numel(testY)