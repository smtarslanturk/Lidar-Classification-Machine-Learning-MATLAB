% Bu kod x,y,z,intensty,returnNumber ve s�n�f olmak �zeri 6 s�t�n dahildir.
% NOT: Classification Learner �al��t�rmadan �nce t�m featureler bir
% matriste toplanmal�d�r.
% NOT: Birle�tirilen matriste s�n�flar en sonda(en sa�daki s�t�nda)
% olmal�d�rki. Classification Learner bunun s�n�f oldu�unu anlas�n.

clc; clear all; close all;
load OMA_198_PC3.txt
load OMA_198_CLS.txt
A = [OMA_198_PC3(1:100:end,1),OMA_198_PC3(1:100:end,2),...
    OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4),OMA_198_PC3(1:100:end,5)];
% bizim i�in �nemli olan bilgiler 3. s�t�n(z) ve 4. s�t�n(intensty) bilgisi
siniflar = OMA_198_CLS(1:100:end); %sadece 0,2,5,6 s�n�flar� mevcut
% cellSiniflar = num2cell(siniflar);

%S�n�f Etiketlerinde 0 olanlar� att�k. Etiketi olmad��� i�in.
idx = siniflar==0;
siniflar = siniflar(~idx);
A = A(~idx,:);


tumVeri = [A siniflar];
[trainedClassifier, validationAccuracy] = trainClassifierWith6Features(tumVeri);
% E�itili�i sa��ndaki yerin ismini fonksiyon adina g�re de�i�tir.


A = [OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),...
    OMA_198_PC3(1:1:end,3),OMA_198_PC3(1:1:end,4),OMA_198_PC3(1:1:end,5)];

[cPredict,score] = predict(trainedClassifier.ClassificationSVM,A);
siniflar = OMA_198_CLS(1:1:end);

figure(1);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],siniflar);
figure(2);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],cPredict);


Accuracy=mean(siniflar==cPredict)*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));



% Ger�ek sifilar� ve cPredict G�sterme 
% 
% disp('class predict');
% disp([siniflar cPredict]);
% 
% %% Ka� Do�rulukla �al��t���


