% Bu kod x,y,z,intensty,returnNumber ve sýnýf olmak üzeri 6 sütün dahildir.
% NOT: Classification Learner çalýþtýrmadan önce tüm featureler bir
% matriste toplanmalýdýr.
% NOT: Birleþtirilen matriste sýnýflar en sonda(en saðdaki sütünda)
% olmalýdýrki. Classification Learner bunun sýnýf olduðunu anlasýn.

clc; clear all; close all;
load OMA_198_PC3.txt
load OMA_198_CLS.txt
A = [OMA_198_PC3(1:100:end,1),OMA_198_PC3(1:100:end,2),...
    OMA_198_PC3(1:100:end,3),OMA_198_PC3(1:100:end,4),OMA_198_PC3(1:100:end,5)];
% bizim için önemli olan bilgiler 3. sütün(z) ve 4. sütün(intensty) bilgisi
siniflar = OMA_198_CLS(1:100:end); %sadece 0,2,5,6 sýnýflarý mevcut
% cellSiniflar = num2cell(siniflar);

%Sýnýf Etiketlerinde 0 olanlarý attýk. Etiketi olmadýðý için.
idx = siniflar==0;
siniflar = siniflar(~idx);
A = A(~idx,:);


tumVeri = [A siniflar];
[trainedClassifier, validationAccuracy] = trainClassifierWith6Features(tumVeri);
% Eþitiliði saðýndaki yerin ismini fonksiyon adina göre deðiþtir.


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



% Gerçek sifilarý ve cPredict Gösterme 
% 
% disp('class predict');
% disp([siniflar cPredict]);
% 
% %% Kaç Doðrulukla Çalýþtýðý


