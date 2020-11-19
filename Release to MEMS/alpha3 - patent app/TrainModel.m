% param unlikely to change
cproxpixel = 30;
digitDatasetPath = [pwd '/Training'];
inputSize = [cproxpixel*2+1 cproxpixel*2+1 1];
numClasses = 2;

% load labels from folder names
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% param
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(15,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train model 'net'
net = trainNetwork(imds,layers,options);
save net;