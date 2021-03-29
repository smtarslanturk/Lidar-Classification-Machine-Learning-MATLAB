clc; clear all; close all;


load JAX_166_PC3.txt
load JAX_166_CLS.txt

veriler = JAX_166_PC3(1:50:end,:);
siniflar = JAX_166_CLS(1:50:end); 

% verilerTrans = [A.' B.' C.']; 
% veriler = verilerTrans.';
% 
% siniflarTrans = [A_siniflar.' B_siniflar.' C_siniflar.']; 
% % siniflar = siniflarTrans.';

idx = siniflar==0;
siniflar = siniflar(~idx);
veriler = veriler(~idx,:);

tumVeri = [veriler(:,3:5) siniflar];

% Bu aþamadan sonra Classification Learner devreye girer.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);
% Eþitiliði saðýndaki yerin ismini fonksiyon adina göre deðiþtir.



yeni = [tumVeri(1:1:end,1),tumVeri(1:1:end,2),tumVeri(1:1:end,3)];
[cPredict,score] = predict(trainedClassifier.ClassificationSVM,yeni);
siniflar = tumVeri(1:1:end,4);

figure(1);
pcshow([tumVeri(1:1:end,1),tumVeri(1:1:end,2),tumVeri(1:1:end,3)],siniflar);
figure(2);
pcshow([tumVeri(1:1:end,1),tumVeri(1:1:end,2),tumVeri(1:1:end,3)],cPredict);


Accuracy=mean(siniflar==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));
