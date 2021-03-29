clear;  clc; close all;
%versicolor setosa viring çiçek türü olarak düþün

load fisheriris
inds = ~strcmp(species,'setosa'); 
%species matrisinde 'setosa' yazanlara 0 gerisine 1 atadýk.
X = meas(inds,3:4);
%inds fonksuyonundaki 50 den sonraki 3 ve 4. stün degerlerini alýr.
%belkide inds fonksiyonundaki 1 olan degerlerin 3 ve 4. satýrý al demek.
y = species(inds);
%inds matrisindeki 'setosa' dýsýndaki sýnýflarý tek sütün olarak yazmýs. 

% Buraya kadar kullanacak olduðumuz deðerleri aldýk.

SVMModel = fitcsvm(X,y); 
%SVM modeli oluþturduk.
% Mdl = fitcsvm(X,Y) returns an SVM classifier trained using the predictors 
% in the matrix X and the class labels in vector Y for one-class or 
% two-class classification.

classOrder = SVMModel.ClassNames;
%2 tane sýnýf kaldýgýný soyledik.
%2 tane class var= 'versicolor' 'virginica'
sv = SVMModel.SupportVectors; 
%Aslýnda svmmodel içince supportvector gibi datalar saklý zaten
%Çekmek istediðimiz bilgileri SVMModel sýnýfýnýn içerisinden çekiyoruz.

figure
gscatter(X(:,1),X(:,2),y,'br','xo');  
% Xe attýðýmýz datalara ve y sýnýfýna göre çizim yapmayý söyledik.
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
%Support vector ile yapýlan çizimdeki support vectorle birleþtirdik.
%ko çizimin þeklini belirler,
%10 dairelerin boyutunu belirler.
legend('versicolor','virginica','Support Vector')
hold off

%% STRCMP fonksiyonu:
s1 = 'Yes';
s2 = 'No';
tf1 = strcmp(s1,s2); %degerler esit olmadýgý ýcýn tf=0

a1='1';
a2='1';
tf2 = strcmp(a1,a2); %ayný karakterler oldugu icin tf2=1

% Find Text in Cell Array. 
% bir array icinde aradýgýmýz bir seyi bulmak istiyorsak.

s1 = 'upon';
s2 = {'Once','upon';
      'a','time'};
tf3 = strcmp(s1,s2); 
% tf3'te sonuc olarak su yazar
% tf3=( 0 1: 0 0), yani sadece 1. satýrýn 2. stununda var demek ýstemýs

s1 = '2';
s2 = {'1','2';
      '1','2'};
tf4 = strcmp(s1,s2); % sonuc olarak ýký tane 1 gelecektýr 0 1 0 1

% Compare Two Cell Arrays of Character Vectors:
s1 = {'Time','flies','when','1';'you','re','having','fun.'};
s2 = {'Time','drags','when','1'; 'you','re','anxiously','waiting.'};
      
tf5 = strcmp(s1,s2);

%% MEAS: aslýnda bu matris içinde belli stünlara verilen isim gibidir. 

load fisheriris;

PL = meas(:,3); %petal length (third column in meas)-sadece 3. stün datalarý aldýk.
PW = meas(:,4); %petal width (fourth column in meas)-sadece 4. stün datalarý aldýk.

h1 = gscatter(PL,PW,species,'krb','ov^',[],'off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best')
%legend best ile acýklama kutusunu bos olan yere yerlestirmis.
hold on

%sonuc olarak ise: setosa icin PW-PL degerleri cok kucuktur. Virginica icin
% ise en yüksek degerler mevcuttur.

%% sv = SVMModel.SupportVectors Komutu
%Aslýnda burda dikkat edilmesi gereken nolta SVMModel Sýnýflandýrmasýnýn 
%dogru bir sekilde yapýlmasýdýr.

load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y = species(inds);

SVMModel = fitcsvm(X,y);
%Aslýnda svmmodel içince supportvector gibi datalar saklý zaten
%Çekmek istediðimiz bilgileri SVMModel sýnýfýnýn içerisinden çekiyoruz.
svl= SVMModel.SupportVectorLabels;
classOrder = SVMModel.ClassNames;
sv = SVMModel.SupportVectors;

figure
gscatter(X(:,1),X(:,2),y)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('versicolor','virginica','Support Vector')
hold off

%% FITCSVM KOMUTU fitcsvm();
%  for one-class and binary classification.
% supports sequential minimal optimization (SMO),
% iterative single data algorithm (ISDA), or L1 soft-margin minimization,
% a low-dimensional or moderate-dimensional predictor data set,
% fitclinear= bir cok parametre varsa buda kullanýlabilir.

% For multiclass learning with combined binary SVM models, use error-correcting output codes (ECOC). 
% For more details, see fitcecoc.

% To train an SVM regression model, see fitrsvm for low-dimensionalmoderate
% -dimensionalpredictor data sets, or fitrlinear for high-dimensional data sets.

% Train SVM Classifier:
% This is a binary classification problem. 
% "b" for bad radar returns and "g" for good radar returns.

load ionosphere
rng(1); % For reproducibility
SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

CVSVMModel = crossval(SVMModel);

classLoss = kfoldLoss(CVSVMModel);
%classLoss denilen degerin bulunmasý saðlanmýþtýr.

%%  GSCATTER Fonksiyonu

load discrim

figure;
gscatter(ratings(:,1),ratings(:,2),group,'br','xo')
xlabel('climate');
ylabel('housing');

% genel olarak plot gibi çizim yapmak amacýyla kullanýlýyor galiba.



