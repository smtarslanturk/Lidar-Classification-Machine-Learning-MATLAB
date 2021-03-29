clc; clear all; close all;
load JAX_280_PC3.txt
load JAX_280_CLS.txt

% Alt �rnekleme yap�la�m. ��lem kolayl��� ve ilerde yap�lacak i�lemlerde
% e�it �artlar olmas� ad�na.
JAX_280_CLS = JAX_280_CLS(1:10:end);
JAX_280_PC3 = JAX_280_PC3(1:10:end,:);

idx = JAX_280_CLS==0;
JAX_280_CLS = JAX_280_CLS(~idx);
JAX_280_PC3 = JAX_280_PC3(~idx,:);

verilerEgitim = JAX_280_PC3(1:50:end,:); 
siniflarEgitim = JAX_280_CLS(1:50:end); 

tumVeri = [verilerEgitim(:,:) siniflarEgitim];
% Classification Learner� �al��t�r�al�m ver t�m veriyi verelim.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);

indis = setdiff (1:length(JAX_280_CLS),1:50:length(JAX_280_CLS));
indis = indis';
% A�a��daki i�lemde indislerde ald��� de�erleri g�sterir.


[cPredict,score] = predict(trainedClassifier.ClassificationKNN,JAX_280_PC3(1:1:end,:));

figure(1);
pcshow(JAX_280_PC3(1:1:end,1:3),JAX_280_CLS);
figure(2);
pcshow(JAX_280_PC3(1:1:end,1:3),cPredict);

siniflarTest = JAX_280_CLS(indis);
siniflarPredict = cPredict(indis);

Accuracy=mean(siniflarTest==siniflarPredict)*100;
Accuracy2 =mean(JAX_280_CLS==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));






