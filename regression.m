clear;
clc;
tic;
load input_noCov.mat;
load output_south.mat;
disp('loaded data');
X = X(1:300,:);
y = y(1:300,:);
initAvg = mean(X);
X = X - repmat( initAvg, size(X,1), 1 );
initRan = 2*std(X);
X = X ./ repmat( initRan, size(X,1), 1 );
X = pca(X,0.999);
disp('performed PCA');
disp(size(X,2));

init_features = size(X,2);
for i = 1:init_features,
    for j = i:init_features,
        X = [ X X(:,i).*X(:,j) ]; 
    end
end

X = [X ones(size(X,1),1)];
disp('created final feature set');
origX = X;
size(origX)
size(y)


for predictionYear = 2011:2012,
    
    Xtest = origX(end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122,:);
    ytest = origX(end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122,:);
    X = origX(1 : end - ( (2012-predictionYear)*122+122 ),:);
    y = origX(1 : end - ( (2012-predictionYear)*122+122 ),:);
    
    mdl = regress(y, X, 'linear');
    disp('completed regression');

    predicted = Xtest * mdl;
    error = mean( mean( abs( predicted - ytest ) * 100 ./ mean(origY) ) );
    disp(['mean absolute percentage error for ' num2str(predictionYear) ' year is ' num2str(error)]);
end

t = toc;
disp(['took ' num2str(t) ' seconds']);