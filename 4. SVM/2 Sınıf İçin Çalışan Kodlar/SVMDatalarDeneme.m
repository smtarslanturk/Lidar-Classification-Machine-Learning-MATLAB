clc; clear all; close all;
load OMA_198_PC3.txt
load OMA_198_CLS.txt
A = [OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4)];
% bizim için önemli olan bilgiler 3. sütün(z) ve 4. sütün(intensty) bilgisi
siniflar = OMA_198_CLS(1:100:end); %sadece 0,2,5,6 sýnýflarý mevcut
% cellSiniflar = num2cell(siniflar);

%Sýnýf Etiketlerinde 0 olanlarý attýk. Etiketi olmadýðý için.
idx = siniflar==0;
siniflar = siniflar(~idx);
A = A(~idx,:);

%Sýnýf etiketlerinden 6'yý attýk.
idx = siniflar==6;
siniflar = siniflar(~idx);
A = A(~idx,:);
%%
SVMModel = fitcsvm(A,siniflar); 
%SVM modeli oluþturduk.
classOrder = SVMModel.ClassNames;
%2 tane sýnýf kaldýgýný soyledik.
%2 tane class var= '2' '5'
sv = SVMModel.SupportVectors; 
% Support vector çektik.

%% Eðitim Kýsmýný tamamladýk artýk tahmin edelim:

A = [OMA_198_PC3(1:1:end,3),OMA_198_PC3(1:1:end,4)];
[cPredict,score] = predict(SVMModel,A);
siniflar = OMA_198_CLS(1:1:end);
figure(1);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],siniflar);
figure(2);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],cPredict);

%% Gerçek sifilarý ve cPredict Gösterme 

disp('class predict');
disp([siniflar cPredict]);

%% Kaç Doðrulukla Çalýþtýðý

Accuracy=mean(siniflar==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));

