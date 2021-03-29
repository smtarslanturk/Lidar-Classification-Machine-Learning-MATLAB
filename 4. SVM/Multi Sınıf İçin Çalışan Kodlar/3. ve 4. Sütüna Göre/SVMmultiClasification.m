clc; clear all; close all;
load JAX_166_PC3.txt
load JAX_166_CLS.txt

verilerEgitim = JAX_166_PC3(1:50:end,:); 
siniflarEgitim = JAX_166_CLS(1:50:end); 

%Sýnýf Etiketlerinde 0 olanlarý attýk. Etiketi olmadýðý için.
idx = siniflarEgitim==0;
siniflarEgitim = siniflarEgitim(~idx);
verilerEgitim = verilerEgitim(~idx,:);


% %% Eðitim Kýsmý

tumVeri = [verilerEgitim(:,3:4) siniflarEgitim];
% Classification Learnerý Çalýþtýrýalým ver tüm veriyi verelim.

[trainedClassifier, validationAccuracy] = trainClassifier(tumVeri);
%Bu kýsýmda çok zaman alýyor ondan her seferinde çalýþtýrma.

% % Test Kýsmý
veri = JAX_166_PC3(1:1:end,:);
siniflar = JAX_166_CLS;
idx = siniflar==0;
veri = veri(~idx,:);
siniflar = siniflar (~idx,:);

% Aþaðýdaki iþlemde indis deðerlerini döndürür.
indis = setdiff (1:length(siniflar),1:50:length(siniflar));
indis = indis';
% Aþaðýdaki iþlemde indislerde aldýðý deðerleri gösterir.
siniflarTest = siniflar(indis);

% A'nýn tüm verilerini aldýk tekrar test için
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

% %% Kaç Doðrulukla Çalýþtýðý
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









