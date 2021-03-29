clc; clear all; close all;
% load JAX_166_PC3.txt
% load JAX_166_CLS.txt

load OMA_124_PC3.txt
load OMA_124_CLS.txt

idx = OMA_124_CLS==0;
OMA_124_CLS = OMA_124_CLS(~idx);
OMA_124_PC3 = OMA_124_PC3(~idx,:);

verilerEgitim = OMA_124_PC3(1:2000:end,:); 
siniflarEgitim = OMA_124_CLS(1:2000:end); 

tumVeri = [verilerEgitim(:,:) siniflarEgitim];
% Classification Learnerý Çalýþtýrýalým ver tüm veriyi verelim.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);

indis = setdiff (1:length(OMA_124_CLS),1:2000:length(OMA_124_CLS));
indis = indis';
% Aþaðýdaki iþlemde indislerde aldýðý deðerleri gösterir.


[cPredict,score] = predict(trainedClassifier.ClassificationSVM,OMA_124_PC3(1:1:end,:));

figure(1);
pcshow(OMA_124_PC3(1:1:end,1:3),OMA_124_CLS);
figure(2);
pcshow(OMA_124_PC3(1:1:end,1:3),cPredict);

siniflarTest = OMA_124_CLS(indis);
siniflarPredict = cPredict(indis);

Accuracy=mean(siniflarTest==siniflarPredict)*100;
Accuracy2 =mean(OMA_124_CLS==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));






