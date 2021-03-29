%Elimizdeki Datalar� G�rselle�tirmek ��in Yaz�lan Kod

clc; clear all; close all;
name = 'OMA_198'; %Dosya ismine g�re name de�i�kenini de�i�tir.

%Dosyan�n bulundugu konuma g�re fileName k�sm�ndan atama yapar�z.
fileName = ...
    ['C:\Users\smtar\Desktop\TEZ 2019 AL�CAN HOCA\3. Matlab �al��malar\1. Tez ��in Denemeler\3. Alican Hoca PCSHOW\'...
    ,name,'_PC3.txt'];
OMA132PC3 = importfile(fileName);

%Dosyan�n konumuna g�re s�n�f etiketini ayarl�yoruz.
fileName2 = ...
    ['C:\Users\smtar\Desktop\TEZ 2019 AL�CAN HOCA\3. Matlab �al��malar\1. Tez ��in Denemeler\3. Alican Hoca PCSHOW\'...
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
%% Intensty G�sterimi ��in Yaz�lan Kod 
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

%% SVM �le S�n�fland�rma: 

% A = [OMA132PC3(1:500,3),OMA132PC3(1:500,4)];
% % bizim i�in �nemli olan bilgiler 3. s�t�n(z) ve 4. s�t�n(intensty) bilgisi
% 
% siniflar = gt(1:500);%sadece 0,2,5,6 s�n�flar� mevcut
% % cellSiniflar = num2cell(siniflar);
% 
% SVMModel = fitcsvm(A,siniflar); 
% %SVM modeli olu�turduk.
% classOrder = SVMModel.ClassNames;
% %2 tane s�n�f kald�g�n� soyledik.
% %2 tane class var= '2' '5'
% sv = SVMModel.SupportVectors; 
% % Support vector �ektik.
% 
% figure();
% gscatter(A(:,1), A(:,2), siniflar,'rgb','osd');
% % figure();
% % gscatter(A(:,1), A(:,2), siniflar,'br','xo');
% %yukar�da farkl� �ekillerde g�sterimler sa�lanm��.
% hold on
% plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
% legend('2','5','Support Vector');
% hold off;
% xlabel('Y�kseklik');
% ylabel('�ntensty');
% N = size(siniflar,1);
% 
% lda = fitcdiscr(A(:,1:2),siniflar);
% ldaClass = resubPredict(lda);
% 
% ldaResubErr = resubLoss(lda);

