clc; clear all; close all;
load JAX_166_PC3.txt
load JAX_166_CLS.txt

JAX_166_CLS = JAX_166_CLS(1:10:end);
JAX_166_PC3 = JAX_166_PC3(1:10:end,:);

idx = JAX_166_CLS==0;
JAX_166_CLS = JAX_166_CLS(~idx);
JAX_166_PC3 = JAX_166_PC3(~idx,:);

verilerEgitim = JAX_166_PC3(1:50:end,:); 
siniflarEgitim = JAX_166_CLS(1:50:end); 

tumVeri = [verilerEgitim(:,:) siniflarEgitim];
% Classification Learnerý Çalýþtýrýalým ver tüm veriyi verelim.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);

indis = setdiff (1:length(JAX_166_CLS),1:50:length(JAX_166_CLS));
indis = indis';
% Aþaðýdaki iþlemde indislerde aldýðý deðerleri gösterir.


[cPredict,score] = predict(trainedClassifier.ClassificationKNN,JAX_166_PC3(1:1:end,:));

figure(1);
pcshow(JAX_166_PC3(1:1:end,1:3),JAX_166_CLS);
figure(2);
pcshow(JAX_166_PC3(1:1:end,1:3),cPredict);

siniflarTest = JAX_166_CLS(indis);
siniflarPredict = cPredict(indis);

Accuracy=mean(siniflarTest==siniflarPredict)*100;
Accuracy2 =mean(JAX_166_CLS==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));






