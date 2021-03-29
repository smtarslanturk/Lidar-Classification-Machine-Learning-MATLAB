clc; clear all; close all;
load JAX_166_PC3.txt
load JAX_166_CLS.txt

verilerEgitim = JAX_166_PC3(1:50:end,:); 
siniflarEgitim = JAX_166_CLS(1:50:end); 

%S�n�f Etiketlerinde 0 olanlar� att�k. Etiketi olmad��� i�in.
idx = siniflarEgitim==0;
siniflarEgitim = siniflarEgitim(~idx);
verilerEgitim = verilerEgitim(~idx,:);


% %% E�itim K�sm�

tumVeri = [verilerEgitim(:,3:4) siniflarEgitim];
% Classification Learner� �al��t�r�al�m ver t�m veriyi verelim.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);
%Bu k�s�mda �ok zaman al�yor ondan her seferinde �al��t�rma.

% % Test K�sm�
veri = JAX_166_PC3(1:1:end,:);
siniflar = JAX_166_CLS;
idx = siniflar==0;
veri = veri(~idx,:);
siniflar = siniflar (~idx,:);

% A�a��daki i�lemde indis de�erlerini d�nd�r�r.
indis = setdiff (1:length(siniflar),1:50:length(siniflar));
indis = indis';
% A�a��daki i�lemde indislerde ald��� de�erleri g�sterir.
siniflarTest = siniflar(indis);

% A'n�n t�m verilerini ald�k tekrar test i�in
[cPredict,score] = predict(trainedClassifier.ClassificationSVM,veri(1:1:end,3:4));
% indis2 = 1:50:length(siniflar);
% indis2 = indis2';
% cPredict = siniflar(indis2);
test2 = cPredict(indis);

figure(1);
pcshow(veri(1:1:end,1:3),siniflar);
figure(2);
pcshow(veri(1:1:end,1:3),cPredict);

Accuracy=(sum(siniflarTest==test2)/length(test2))*100;
fprintf('\nAccuracy =%d\n',uint8(Accuracy));

% %% Ka� Do�rulukla �al��t���
% Accuracy=mean(siniflar==cPredict)*100;
% Accuracy=(sum(siniflar==cPredict)/length(cPredict))*100;
% fprintf('\nAccuracy =%d\n',uint8(Accuracy));

% indis = setdiff (1:length(siniflar),1:50:length(siniflar));
% test = siniflar(indis);
% test2 = cPredict(indis);
% 
% siniflar = siniflar (~idx,:);
% 
% 
% Accuracy3=(sum(test==test2)/length(test2))*100;
% 


% r3 = (1:1:1000);
% r3 = reshape(r3,5,200);
% r3 = r3';
% [k,l] = size(r3);
% rastgele = r3(1:10:end,:);
% [m,n] = size(rastgele);
% diger = setdiff(r3,rastgele);
% digerMatris = reshape(diger,5,k-m);
% kalan = digerMatris';









