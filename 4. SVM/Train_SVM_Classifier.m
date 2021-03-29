clc; clear all; close all;

load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y = species(inds);

SVMModel = fitcsvm(X,y);
classOrder = SVMModel.ClassNames;

sv = SVMModel.SupportVectors;
figure(1);
gscatter(X(:,1),X(:,2),y,'br','xo');  %kýrmýzý ve mavi noktalarý gösterir.
hold on;
plot(sv(:,1),sv(:,2),'ko','MarkerSize',15);
legend('versicolor','virginica','Support Vector');
hold off;

CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);

%%
clc; clear all; close all;

load ionosphere;
rng(1); % For reproducibility

SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

CVSVMModel = crossval(SVMModel);

classLoss = kfoldLoss(CVSVMModel);

%% Detect Outliers Using SVM and One-Class Learning
clc; clear all; close all;

load fisheriris;
X = meas(:,1:2);
y = ones(size(X,1),1);

% rng(1);
SVMModel = fitcsvm(X,y,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.05);
svInd = SVMModel.IsSupportVector;
h = 0.02; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)),...
    min(X(:,2)):h:max(X(:,2)));
[~,score] = predict(SVMModel,[X1(:),X2(:)]);
scoreGrid = reshape(score,size(X1,1),size(X2,2));

figure(3);
plot(X(:,1),X(:,2),'k.');
hold on;
plot(X(svInd,1),X(svInd,2),'ro','MarkerSize',10);
contour(X1,X2,scoreGrid);
colorbar;
title('{\bf Iris Outlier Detection via One-Class SVM}');
xlabel('Sepal Length (cm)');
ylabel('Sepal Width (cm)');
legend('Observation','Support Vector');
hold off;

CVSVMModel = crossval(SVMModel);
[~,scorePred] = kfoldPredict(CVSVMModel);
outlierRate = mean(scorePred<0);

%% Find Multiple Class Boundaries Using Binary SVM
clc; clear all; close all;

load ('fisheriris');
X = meas(:,3:4);
Y = species;

figure();
gscatter(X(:,1),X(:,2),Y);
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Scatter Diagram of Iris Measurements}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend('Location','Northwest');

SVMModels = cell(3,1);
classes = unique(Y);
rng(1); % For reproducibility

for j = 1:numel(classes)
    indx = strcmp(Y,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end

d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

figure ();
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
hold on
h(4:6) = gscatter(X(:,1),X(:,2),Y);
title('{\bf Iris Classification Regions}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend(h,{'setosa region','versicolor region','virginica region',...
    'observed setosa','observed versicolor','observed virginica'},...
    'Location','Northwest');
axis tight
hold off