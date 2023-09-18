

% Clear workspace
clear; close all; clc;
% Images Datapath – Please modify your path accordingly 
datapath='D:\COVID-19_Radiography_Dataset\Processed_Images';





% Image Datastore
imds=imageDatastore(datapath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Determine the split up
[TrainImages, TestImages] = splitEachLabel(imds, 0.8, 'randomized');

% Define the layers
layers = [
    imageInputLayer([256 256 1], 'Name', 'Input')
    
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'Conv_1')
    batchNormalizationLayer('Name', 'BN_1')
    reluLayer('Name', 'Relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_1')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'Conv_2')
    batchNormalizationLayer('Name', 'Bn_2')
    reluLayer('Name', 'Relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_2')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'Conv_3')
    batchNormalizationLayer('Name', 'Bn_3')
    reluLayer('Name', 'Relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_3')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv_4')
    batchNormalizationLayer('Name', 'Bn_4')
    reluLayer('Name', 'Relu_4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_4')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'Conv_5')
    batchNormalizationLayer('Name', 'Bn_5')
    reluLayer('Name', 'Relu_5')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_5')
    
    convolution2dLayer(3,256, 'Padding', 'same', 'Name', 'Conv_6')
    batchNormalizationLayer('Name', 'Bn_6')
    reluLayer('Name', 'Relu_6')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_6')
    
    
    fullyConnectedLayer(3, 'Name', 'FC')
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output_Classification')
];

% Create a layer graph and plot it
lgraph = layerGraph(layers);
plot(lgraph);

% Set the training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', TestImages, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
% Train the network
net = trainNetwork(TrainImages, layers, options);

% Make predictions on the test set
YPred = classify(net, TestImages);
YValidation = TestImages.Labels;

% Calculate the accuracy
accuracy = sum(YPred == YValidation) / numel(YValidation);
disp("Test Accuracy: " + accuracy);