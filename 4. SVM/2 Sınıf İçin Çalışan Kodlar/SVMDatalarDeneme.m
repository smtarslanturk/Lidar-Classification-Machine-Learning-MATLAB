clc; clear all; close all;
load OMA_198_PC3.txt
load OMA_198_CLS.txt
A = [OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4)];
% bizim i�in �nemli olan bilgiler 3. s�t�n(z) ve 4. s�t�n(intensty) bilgisi
siniflar = OMA_198_CLS(1:100:end); %sadece 0,2,5,6 s�n�flar� mevcut
% cellSiniflar = num2cell(siniflar);

%S�n�f Etiketlerinde 0 olanlar� att�k. Etiketi olmad��� i�in.
idx = siniflar==0;
siniflar = siniflar(~idx);
A = A(~idx,:);

%S�n�f etiketlerinden 6'y� att�k.
idx = siniflar==6;
siniflar = siniflar(~idx);
A = A(~idx,:);
%%
SVMModel = fitcsvm(A,siniflar); 
%SVM modeli olu�turduk.
classOrder = SVMModel.ClassNames;
%2 tane s�n�f kald�g�n� soyledik.
%2 tane class var= '2' '5'
sv = SVMModel.SupportVectors; 
% Support vector �ektik.

%% E�itim K�sm�n� tamamlad�k art�k tahmin edelim:

A = [OMA_198_PC3(1:1:end,3),OMA_198_PC3(1:1:end,4)];
[cPredict,score] = predict(SVMModel,A);
siniflar = OMA_198_CLS(1:1:end);
figure(1);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],siniflar);
figure(2);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],cPredict);

%% Ger�ek sifilar� ve cPredict G�sterme 

disp('class predict');
disp([siniflar cPredict]);

%% Ka� Do�rulukla �al��t���

Accuracy=mean(siniflar==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));

