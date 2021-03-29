% 3 ayrý dosya üzerinden farklý sýnýflar seçerek iþlem yapmaya çalýþcaðýz.

clc; clear all; close all;

%Kullanýlacak dosyalarýn hem PC hem CLS dosyalarýný yükleyelim.
load OMA_198_PC3.txt
load OMA_198_CLS.txt
load JAX_166_PC3.txt
load JAX_166_CLS.txt
load OMA_124_PC3.txt
load OMA_124_CLS.txt

% A = [OMA_198_PC3(1:500:end,1),OMA_198_PC3(1:500:end,2),OMA_198_PC3(1:500:end,3),...
%     OMA_198_PC3(1:500:end,4),OMA_198_PC3(1:500:end,5)];
% A_siniflar = OMA_198_CLS(1:500:end); 
% B = [JAX_166_PC3(1:500:end,1),JAX_166_PC3(1:500:end,2),JAX_166_PC3(1:500:end,3),...
%     JAX_166_PC3(1:500:end,4),JAX_166_PC3(1:500:end,5)];
% B_siniflar = JAX_166_CLS(1:500:end); 
% C = [OMA_124_PC3(1:500:end,1),OMA_124_PC3(1:500:end,2),OMA_124_PC3(1:500:end,3),...
%     OMA_124_PC3(1:500:end,4),OMA_124_PC3(1:500:end,5)];
% C_siniflar = OMA_124_CLS(1:500:end); 

A = OMA_198_PC3(1:1000:end,:);
A_siniflar = OMA_198_CLS(1:1000:end);
B = JAX_166_PC3(1:1000:end,:);
B_siniflar = JAX_166_CLS(1:1000:end); 
C = OMA_124_PC3(1:1000:end,:);
C_siniflar = OMA_124_CLS(1:1000:end);

% korA = [OMA_198_PC3(1:500:end,1),OMA_198_PC3(1:500:end,2),OMA_198_PC3(1:500:end,3)];
% korB = [JAX_166_PC3(1:500:end,1),JAX_166_PC3(1:500:end,2),JAX_166_PC3(1:500:end,3)];
% korC = [OMA_124_PC3(1:500:end,1),OMA_124_PC3(1:500:end,2),OMA_124_PC3(1:500:end,3)];

% sizeA = size(A);
% sizeB = size(B);
% sizeC = size(C);
% size(veriler) = [4756,3];

% Birleþtirme iþleminde satýr sayýlarý uyþmasý için önce transpoze aldýk ve
% ardýndan birleþtirme iþlemi yapýlýp, birleþtirilen datalarýn transpozu
% alýnarak, birleþtirme iþlemi tamamlanmýþtýr.
verilerTrans = [A.' B.' C.']; %Transpozu alýndý
veriler = verilerTrans.';

siniflarTrans = [A_siniflar.' B_siniflar.' C_siniflar.']; %transpoz alýndý
siniflar = siniflarTrans.';

%Sýnýf Etiketlerinde 0 olanlarý attýk. Etiketi olmadýðý için.
idx = siniflar==0;
siniflar = siniflar(~idx);
veriler = veriler(~idx,:);


% kullanýlacak özellikleri birleþtirelim
tumVeri = [veriler(:,3:5) siniflar];

% Bu aþamadan sonra Classification Learner devreye girer.
%Eðitim kýsmý olarak belirtebiliriz:

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);
% Eþitiliði saðýndaki yerin ismini fonksiyon adina göre deðiþtir.


%Lets Start Test Steps:

%Test 1: OMA_198_PC3 dosyasýnýn test iþlemi:
predictA = [OMA_198_PC3(1:1:end,3),OMA_198_PC3(1:1:end,4),OMA_198_PC3(1:1:end,5)];
[cPredict,score] = predict(trainedClassifier.ClassificationSVM,predictA);
siniflar1 = OMA_198_CLS(1:1:end);

figure(1);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],siniflar1);
figure(2);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],cPredict);


Accuracy=mean(siniflar1==cPredict)*100;
fprintf('\nAccuracy for OMA_198 =%d\n',uint8(Accuracy));


%Test 2: JAX_166_PC3 dosyasýnýn test iþlemi:
predictB = [JAX_166_PC3(1:1:end,3),JAX_166_PC3(1:1:end,4),JAX_166_PC3(1:1:end,5)];
[cPredict,score] = predict(trainedClassifier.ClassificationSVM,predictB);
siniflar2 = JAX_166_CLS(1:1:end);

figure(3);
pcshow([JAX_166_PC3(1:1:end,1),JAX_166_PC3(1:1:end,2),JAX_166_PC3(1:1:end,3)],siniflar2);
figure(4);
pcshow([JAX_166_PC3(1:1:end,1),JAX_166_PC3(1:1:end,2),JAX_166_PC3(1:1:end,3)],cPredict);


Accuracy=mean(siniflar2==cPredict)*100;
fprintf('\nAccuracy for JAX_166_PC3 =%d\n',uint8(Accuracy));

%Test 3: OMA_124_PC3 dosyasýnýn test iþlemi:
predictC = [OMA_124_PC3(1:1:end,3),OMA_124_PC3(1:1:end,4),OMA_124_PC3(1:1:end,5)];
[cPredict,score] = predict(trainedClassifier.ClassificationSVM,predictC);
siniflar3 = OMA_124_CLS(1:1:end);

figure(5);
pcshow([OMA_124_PC3(1:1:end,1),OMA_124_PC3(1:1:end,2),OMA_124_PC3(1:1:end,3)],siniflar3);
figure(6);
pcshow([OMA_124_PC3(1:1:end,1),OMA_124_PC3(1:1:end,2),OMA_124_PC3(1:1:end,3)],cPredict);


Accuracy=mean(siniflar3==cPredict)*100;
fprintf('\nAccuracy for OMA_124_PC3 =%d\n',uint8(Accuracy));

% Gerçek sifilarý ve cPredict Gösterme 
% 
% disp('class predict');
% disp([siniflar cPredict]);
% 
% %% Kaç Doðrulukla Çalýþtýðý


