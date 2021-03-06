clc; clear all; close all;
load JAX_166_PC3.txt
load JAX_166_CLS.txt

% Alt ?rnekleme yap?la?m. ??lem kolayl??? ve ilerde yap?lacak i?lemlerde
% e?it ?artlar olmas? ad?na.
JAX_166_CLS = JAX_166_CLS(1:5:end);
JAX_166_PC3 = JAX_166_PC3(1:5:end,:);

idx = JAX_166_CLS==0;
JAX_166_CLS = JAX_166_CLS(~idx);
JAX_166_PC3 = JAX_166_PC3(~idx,:);

verilerEgitim = JAX_166_PC3(1:100:end,:); 
siniflarEgitim = JAX_166_CLS(1:100:end); 

tumVeri = [verilerEgitim(:,3:5) siniflarEgitim];
% Classification Learner? ?al??t?r?al?m ver t?m veriyi verelim.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);

indis = setdiff (1:length(JAX_166_CLS),1:100:length(JAX_166_CLS));
indis = indis';
% A?a??daki i?lemde indislerde ald??? de?erleri g?sterir.


[cPredict,score] = predict(trainedClassifier.ClassificationKNN,JAX_166_PC3(1:1:end,3:5));

% figure(1);
% pcshow(JAX_166_PC3(1:1:end,1:3),JAX_166_CLS);
% figure(2);
% pcshow(JAX_166_PC3(1:1:end,1:3),cPredict);

siniflarTest = JAX_166_CLS(indis);
siniflarPredict = cPredict(indis);

Accuracy=mean(siniflarTest==siniflarPredict)*100;
Accuracy2 =mean(JAX_166_CLS==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));






