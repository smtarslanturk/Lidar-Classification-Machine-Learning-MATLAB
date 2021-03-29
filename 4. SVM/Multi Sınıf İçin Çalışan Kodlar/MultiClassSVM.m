clc; clear all; close all;
load OMA_198_PC3.txt
load OMA_198_CLS.txt
A = [OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4),OMA_198_PC3(1:100:end,5)];
% bizim i�in �nemli olan bilgiler 3. s�t�n(z) ve 4. s�t�n(intensty) bilgisi
siniflar = OMA_198_CLS(1:100:end); %sadece 0,2,5,6 s�n�flar� mevcut
% cellSiniflar = num2cell(siniflar);

%S�n�f Etiketlerinde 0 olanlar� att�k. Etiketi olmad��� i�in.
idx = siniflar==0;
siniflar = siniflar(~idx);
A = A(~idx,:);

tumVeri = [A siniflar];
%Kullan�lacak olan verilerin hepsini bir matrise att�k.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);
% fonksiyonu �al��t�rd�k ve bunun i�inden tahmin yapacaz.

A = [OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4),OMA_198_PC3(1:100:end,5)];
[cPredict,score] = predict(trainedClassifier.ClassificationSVM,A);

siniflar = OMA_198_CLS(1:1:end);
figure(1);
pcshow(...
    [OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4),OMA_198_PC3(1:100:end,5)]...
    ,siniflar);
figure(2);
pcshow(...
    [OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4),OMA_198_PC3(1:100:end,5)]...
    ,cPredict);







% %S�n�f etiketlerinden 6'y� att�k.
% idx = siniflar==6;
% siniflar = siniflar(~idx);
% A = A(~idx,:);
%%
SVMModel = fitcecoc(A,siniflar); 
%SVM modeli olu�turduk.



% %% E�itim K�sm�n� tamamlad�k art�k tahmin edelim:
% 
% tumVeri = [A siniflar];
% %[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);
% %Bu k�s�mda �ok zaman al�yor ondan her seferinde �al��t�rma.
% 
% A = [OMA_198_PC3(1:1:end,3),OMA_198_PC3(1:1:end,4)];
% [cPredict,score] = predict(trainedClassifier.ClassificationSVM,A);

% siniflar = OMA_198_CLS(1:1:end);
% figure(1);
% pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],siniflar);
% figure(2);
% pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],cPredict);
% 
% %% Ger�ek sifilar� ve cPredict G�sterme 
% % 
% % disp('class predict');
% % disp([siniflar cPredict]);
% % 
% % %% Ka� Do�rulukla �al��t���
% Accuracy=mean(siniflar==cPredict)*100;
% fprintf('\nAccuracy =%d\n',uint8(Accuracy));

