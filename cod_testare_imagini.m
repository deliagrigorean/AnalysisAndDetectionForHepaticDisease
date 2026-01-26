load('modelul_antrenat.mat', 'net');

testFolder = 'D:\facultate\ANUL III\SEMESTRUL II\Ingineria reglarii automate II\Proiect\test';  
testDS = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imageSize = [224, 224];
testDS.ReadFcn = @(filename) imresize(imread(filename), imageSize);

predictedLabels = classify(net, testDS);
 
for i = 1:numel(testDS.Files)
    img = readimage(testDS, i);
    imshow(img);
    title(['Etichetă prezisă: ', char(predictedLabels(i))], 'FontSize', 14);
    pause(1);  
end

if ~isempty(testDS.Labels)
    actualLabels = testDS.Labels;

    accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);
    disp(['Acuratețea pe setul de test este: ', num2str(accuracy)]);
    
    figure;
    confusionchart(actualLabels, predictedLabels);
    title('Matricea de confuzie - Set Test');
end
 
predictedLabelsStr = cellstr(predictedLabels);
fileNames = testDS.Files;
resultsTable = table(fileNames, predictedLabelsStr, 'VariableNames', {'Fisier', 'EtichetaPrezisa'});
writetable(resultsTable, 'rezultate_test.csv');

