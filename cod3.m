trainFolder = 'D:\facultate\ANUL III\SEMESTRUL II\Ingineria reglarii automate II\Proiect\train';
validFolder = 'D:\facultate\ANUL III\SEMESTRUL II\Ingineria reglarii automate II\Proiect\valid';

classNames = {'cancer-coronal', 'cancer-sagittal', 'cancer-transverse', 'normal-coronal', 'normal-sagittal', 'normal-transverse'};

trainDS = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
validDS = imageDatastore(validFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imageSize = [224, 224];
trainDS.ReadFcn = @(filename)imresize(imread(filename), imageSize);
validDS.ReadFcn = @(filename)imresize(imread(filename), imageSize);

layers = [
    imageInputLayer([224 224 3], 'Name', 'input', 'Normalization', 'none') 

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1') 
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1') 

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2') 
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2') 

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3') 
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3') 
   
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout1')
    
    fullyConnectedLayer(6, 'Name', 'fc2')  
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-4, ...  
    'MaxEpochs', 10, ...  
    'Shuffle', 'every-epoch', ...
    'ValidationData', validDS, ...  
    'ValidationFrequency', 30, ...  
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(trainDS, layers, options);

predictedLabels = classify(net, validDS);
actualLabels = validDS.Labels;

accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);
disp(['Accuratețea pe setul de validare este: ', num2str(accuracy)]);

save('modelul_antrenat.mat', 'net');

