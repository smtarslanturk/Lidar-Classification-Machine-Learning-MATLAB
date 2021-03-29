% 3 ayr� dosya �zerinden farkl� s�n�flar se�erek i�lem yapmaya �al��ca��z.

clc; clear all; close all;

%Kullan�lacak dosyalar�n hem PC hem CLS dosyalar�n� y�kleyelim.
load OMA_198_PC3.txt
load OMA_198_CLS.txt
load JAX_166_PC3.txt
load JAX_166_CLS.txt
load OMA_124_PC3.txt
load OMA_124_CLS.txt

% �nceki yap�lan i�lemler de se�mlerin cok dengesi oldu�u g�r�ld� bundan
% dolay� ad�m aral�klar�n� esit almadan denemek istiyoruz.
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

% Birle�tirme i�leminde sat�r say�lar� uy�mas� i�in �nce transpoze ald�k ve
% ard�ndan birle�tirme i�lemi yap�l�p, birle�tirilen datalar�n transpozu
% al�narak, birle�tirme i�lemi tamamlanm��t�r.
verilerTrans = [A.' B.' C.']; %Transpozu al�nd�
veriler = verilerTrans.';

siniflarTrans = [A_siniflar.' B_siniflar.' C_siniflar.']; %transpoz al�nd�
siniflar = siniflarTrans.';

%S�n�f Etiketlerinde 0 olanlar� att�k. Etiketi olmad��� i�in.
idx = siniflar==0;
siniflar = siniflar(~idx);
veriler = veriler(~idx,:);


% kullan�lacak �zellikleri birle�tirelim
% Kulan�lan �zellikler: 1 2 3 4 5. s�t�n ve s�n�flar
tumVeri = [veriler siniflar];

% Bu a�amadan sonra Classification Learner devreye girer.
%E�itim k�sm� olarak belirtebiliriz:

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);
% E�itili�i sa��ndaki yerin ismini fonksiyon adina g�re de�i�tir.


%Lets Start Test Steps:

%Test 1: OMA_198_PC3 dosyas�n�n test i�lemi:
predictA = [OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3),...
OMA_198_PC3(1:1:end,4),OMA_198_PC3(1:1:end,5)];
[cPredict,~] = predict(trainedClassifier.ClassificationSVM,predictA);
siniflar = OMA_198_CLS(1:1:end);

figure(1);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],siniflar);
figure(2);
pcshow([OMA_198_PC3(1:1:end,1),OMA_198_PC3(1:1:end,2),OMA_198_PC3(1:1:end,3)],cPredict);


Accuracy=mean(siniflar==cPredict)*100;
fprintf('\nAccuracy for OMA_198 =%d\n',uint8(Accuracy));


%Test 2: JAX_166_PC3 dosyas�n�n test i�lemi:
predictB = [JAX_166_PC3(1:1:end,1),JAX_166_PC3(1:1:end,2),JAX_166_PC3(1:1:end,3),...
JAX_166_PC3(1:1:end,4),JAX_166_PC3(1:1:end,5)];
[cPredict,~] = predict(trainedClassifier.ClassificationSVM,predictB);
siniflar = JAX_166_CLS(1:1:end);

figure(3);
pcshow([JAX_166_PC3(1:1:end,1),JAX_166_PC3(1:1:end,2),JAX_166_PC3(1:1:end,3)],siniflar);
figure(4);
pcshow([JAX_166_PC3(1:1:end,1),JAX_166_PC3(1:1:end,2),JAX_166_PC3(1:1:end,3)],cPredict);


Accuracy=mean(siniflar==cPredict)*100;
fprintf('\nAccuracy for JAX_166_PC3 =%d\n',uint8(Accuracy));

%Test 3: OMA_124_PC3 dosyas�n�n test i�lemi:
predictC = [OMA_124_PC3(1:1:end,1),OMA_124_PC3(1:1:end,2),OMA_124_PC3(1:1:end,3),...
OMA_124_PC3(1:1:end,4),OMA_124_PC3(1:1:end,5)];
[cPredict,score] = predict(trainedClassifier.ClassificationSVM,predictC);
siniflar = OMA_124_CLS(1:1:end);

figure(5);
pcshow([OMA_124_PC3(1:1:end,1),OMA_124_PC3(1:1:end,2),OMA_124_PC3(1:1:end,3)],siniflar);
figure(6);
pcshow([OMA_124_PC3(1:1:end,1),OMA_124_PC3(1:1:end,2),OMA_124_PC3(1:1:end,3)],cPredict);


Accuracy=mean(siniflar==cPredict)*100;
fprintf('\nAccuracy for OMA_124_PC3 =%d\n',uint8(Accuracy));

% Ger�ek sifilar� ve cPredict G�sterme 
% 
% disp('class predict');
% disp([siniflar cPredict]);
% 
% %% Ka� Do�rulukla �al��t���



