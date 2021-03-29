clear;  clc; close all;
%versicolor setosa viring �i�ek t�r� olarak d���n

load fisheriris
inds = ~strcmp(species,'setosa'); 
%species matrisinde 'setosa' yazanlara 0 gerisine 1 atad�k.
X = meas(inds,3:4);
%inds fonksuyonundaki 50 den sonraki 3 ve 4. st�n degerlerini al�r.
%belkide inds fonksiyonundaki 1 olan degerlerin 3 ve 4. sat�r� al demek.
y = species(inds);
%inds matrisindeki 'setosa' d�s�ndaki s�n�flar� tek s�t�n olarak yazm�s. 

% Buraya kadar kullanacak oldu�umuz de�erleri ald�k.

SVMModel = fitcsvm(X,y); 
%SVM modeli olu�turduk.
% Mdl = fitcsvm(X,Y) returns an SVM classifier trained using the predictors 
% in the matrix X and the class labels in vector Y for one-class or 
% two-class classification.

classOrder = SVMModel.ClassNames;
%2 tane s�n�f kald�g�n� soyledik.
%2 tane class var= 'versicolor' 'virginica'
sv = SVMModel.SupportVectors; 
%Asl�nda svmmodel i�ince supportvector gibi datalar sakl� zaten
%�ekmek istedi�imiz bilgileri SVMModel s�n�f�n�n i�erisinden �ekiyoruz.

figure
gscatter(X(:,1),X(:,2),y,'br','xo');  
% Xe att���m�z datalara ve y s�n�f�na g�re �izim yapmay� s�yledik.
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
%Support vector ile yap�lan �izimdeki support vectorle birle�tirdik.
%ko �izimin �eklini belirler,
%10 dairelerin boyutunu belirler.
legend('versicolor','virginica','Support Vector')
hold off

%% STRCMP fonksiyonu:
s1 = 'Yes';
s2 = 'No';
tf1 = strcmp(s1,s2); %degerler esit olmad�g� �c�n tf=0

a1='1';
a2='1';
tf2 = strcmp(a1,a2); %ayn� karakterler oldugu icin tf2=1

% Find Text in Cell Array. 
% bir array icinde arad�g�m�z bir seyi bulmak istiyorsak.

s1 = 'upon';
s2 = {'Once','upon';
      'a','time'};
tf3 = strcmp(s1,s2); 
% tf3'te sonuc olarak su yazar
% tf3=( 0 1: 0 0), yani sadece 1. sat�r�n 2. stununda var demek �stem�s

s1 = '2';
s2 = {'1','2';
      '1','2'};
tf4 = strcmp(s1,s2); % sonuc olarak �k� tane 1 gelecekt�r 0 1 0 1

% Compare Two Cell Arrays of Character Vectors:
s1 = {'Time','flies','when','1';'you','re','having','fun.'};
s2 = {'Time','drags','when','1'; 'you','re','anxiously','waiting.'};
      
tf5 = strcmp(s1,s2);

%% MEAS: asl�nda bu matris i�inde belli st�nlara verilen isim gibidir. 

load fisheriris;

PL = meas(:,3); %petal length (third column in meas)-sadece 3. st�n datalar� ald�k.
PW = meas(:,4); %petal width (fourth column in meas)-sadece 4. st�n datalar� ald�k.

h1 = gscatter(PL,PW,species,'krb','ov^',[],'off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best')
%legend best ile ac�klama kutusunu bos olan yere yerlestirmis.
hold on

%sonuc olarak ise: setosa icin PW-PL degerleri cok kucuktur. Virginica icin
% ise en y�ksek degerler mevcuttur.

%% sv = SVMModel.SupportVectors Komutu
%Asl�nda burda dikkat edilmesi gereken nolta SVMModel S�n�fland�rmas�n�n 
%dogru bir sekilde yap�lmas�d�r.

load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y = species(inds);

SVMModel = fitcsvm(X,y);
%Asl�nda svmmodel i�ince supportvector gibi datalar sakl� zaten
%�ekmek istedi�imiz bilgileri SVMModel s�n�f�n�n i�erisinden �ekiyoruz.
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
% fitclinear= bir cok parametre varsa buda kullan�labilir.

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
%classLoss denilen degerin bulunmas� sa�lanm��t�r.

%%  GSCATTER Fonksiyonu

load discrim

figure;
gscatter(ratings(:,1),ratings(:,2),group,'br','xo')
xlabel('climate');
ylabel('housing');

% genel olarak plot gibi �izim yapmak amac�yla kullan�l�yor galiba.



