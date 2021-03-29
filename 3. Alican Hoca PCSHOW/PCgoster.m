%Elimizdeki Datalarý Görselleþtirmek Ýçin Yazýlan Kod

clc; clear all; close all;
name = 'OMA_198'; %Dosya ismine göre name deðiþkenini deðiþtir.

%Dosyanýn bulundugu konuma göre fileName kýsmýndan atama yaparýz.
fileName = ...
    ['C:\Users\smtar\Desktop\TEZ 2019 ALÝCAN HOCA\3. Matlab Çalýþmalar\1. Tez Ýçin Denemeler\3. Alican Hoca PCSHOW\'...
    ,name,'_PC3.txt'];
OMA132PC3 = importfile(fileName);

%Dosyanýn konumuna göre sýnýf etiketini ayarlýyoruz.
fileName2 = ...
    ['C:\Users\smtar\Desktop\TEZ 2019 ALÝCAN HOCA\3. Matlab Çalýþmalar\1. Tez Ýçin Denemeler\3. Alican Hoca PCSHOW\'...
    ,name,'_CLS.txt'];
gt = importfile1(fileName2);
figure(1);
pcshow([OMA132PC3(:,1),OMA132PC3(:,2),OMA132PC3(:,3)]);
title('acording to X, Y, Z points');
figure(2);
pcshow([OMA132PC3(:,1),OMA132PC3(:,2),OMA132PC3(:,3)],OMA132PC3(:,4));
title('acording to X, Y, Z and *INTENSITY* points');
figure(3);
pcshow([OMA132PC3(:,1),OMA132PC3(:,2),OMA132PC3(:,3)],gt);
title('acording to X, Y, Z and *CLASS* points');

% imshow([OMA132PC3(:,1)/256,OMA132PC3(:,2)/256,OMA132PC3(:,4)/256]);
%% Intensty Gösterimi Ýçin Yazýlan Kod 
intensty = uint8(OMA132PC3(:,4));

yLer = round (OMA132PC3(:,1) + 258);
xLer = round (OMA132PC3(:,2) + 258);

intensityIm = zeros (513,514);

for i = 1: length(xLer)

    intensityIm (yLer(i),xLer(i)) = intensty(i);
    
end

% A = reshape(intensty,256,256);
figure(3);
imshow(intensityIm,[]);
% 0=black 1=white 
title('acording to *INTENSTY* points');

%% SVM Ýle Sýnýflandýrma: 

% A = [OMA132PC3(1:500,3),OMA132PC3(1:500,4)];
% % bizim için önemli olan bilgiler 3. sütün(z) ve 4. sütün(intensty) bilgisi
% 
% siniflar = gt(1:500);%sadece 0,2,5,6 sýnýflarý mevcut
% % cellSiniflar = num2cell(siniflar);
% 
% SVMModel = fitcsvm(A,siniflar); 
% %SVM modeli oluþturduk.
% classOrder = SVMModel.ClassNames;
% %2 tane sýnýf kaldýgýný soyledik.
% %2 tane class var= '2' '5'
% sv = SVMModel.SupportVectors; 
% % Support vector çektik.
% 
% figure();
% gscatter(A(:,1), A(:,2), siniflar,'rgb','osd');
% % figure();
% % gscatter(A(:,1), A(:,2), siniflar,'br','xo');
% %yukarýda farklý þekillerde gösterimler saðlanmýþ.
% hold on
% plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
% legend('2','5','Support Vector');
% hold off;
% xlabel('Yükseklik');
% ylabel('Ýntensty');
% N = size(siniflar,1);
% 
% lda = fitcdiscr(A(:,1:2),siniflar);
% ldaClass = resubPredict(lda);
% 
% ldaResubErr = resubLoss(lda);

