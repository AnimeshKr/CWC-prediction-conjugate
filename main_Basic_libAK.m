clear all;
clc;

% [X,y]=data_extract();
load input_noCov.mat;
load output_south.mat;

%     pos = randperm(size(X,1));
%     X = X(pos,:);y=y(pos,:);

%%%%normalizing for pca%%%%
initAvg = mean(X);
X = X - repmat(initAvg, size(X,1), 1);
initRan = sqrt( mean(X.^2) );
X = X ./ repmat(initRan, size(X,1), 1);
[X, U, error, ~] = pca(X,0.9999);

%%%Normalizing after pca%%%%
ran1 = 3.25*sqrt( var(X) );
X = X ./ repmat(ran1, size(X,1), 1);
y=100*y;
origX=X;
origY=y;
     
trainSize = floor(0.85*size(X,1));
Xval = X(trainSize+1:end,:);
curYearX = X(1:trainSize,:);
yval = y(trainSize+1:end,:);
curYearY = y(1:trainSize,:);


input_layer_size = size(X,2);
hidden_layer_size = 100;
num_labels = size(y,2);
lambda = 0;

startYear = 2005;
endYear = 2012;
prevYears = 110;
nn{1} = 1
nn{2} = 2
nn{3} = 3
for i = 1:(endYear-startYear)+1,

    predictionYear = startYear + i - 1;
    
    %%%%Dividing into validation set, train set and test set%%%%
    Xval    = origX( end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , :);
    yval   = origY( end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , : );

%     curYearX = origX( end - ( ( (2012+prevYears)-predictionYear )*122+121 ) : end - ( (2012-predictionYear)*122+122 ), : );
%     curYearY = origY( end - ( ( (2012+prevYears)-predictionYear )*122+121 ) :end - ( (2012-predictionYear)*122+122 ), : );
  
    curYearX = origX( 1 : end - ( (2012-predictionYear)*122+122 ), : );
    curYearY = origY( 1 : end - ( (2012-predictionYear)*122+122 ), : );

    parfor exp = 1:3,
        
        initial_Theta1 = ( rand( hidden_layer_size, input_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+input_layer_size)) ;
        initial_Theta2 = ( rand( size(y,2), hidden_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+size(y,2))) ;
        initial_nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;

        costFunc = @(p) costFunction(p, ...
                                           input_layer_size, ...
                                           hidden_layer_size, ...
                                           num_labels, curYearX, curYearY, lambda);

        valcostFunc = @(p) costFunction(p, ...
                                           input_layer_size, ...
                                           hidden_layer_size, ...
                                           num_labels, Xval, yval, 0);


        % Now, costFunction is a function that takes in only one argument (the
        % neural network parameters)
        options = optimset('MaxIter', 75);
        [nn_params, cost] = fmincg(costFunc, initial_nn_params, options, valcostFunc,0);

        % Obtain Theta1 and Theta2 back from nn_params
        Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                         hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                         num_labels, (hidden_layer_size + 1));


        nn{exp}.Theta1 = Theta1;
        nn{exp}.Theta2 = Theta2;
        [ trainError, errors(i,exp) ] = feedForward(nn{exp}, X, y, Xval, yval, mean([y; yval]) );

        disp('trainError')
        disp(trainError);
        disp('testError')
        disp(errors(i,exp));
    end
end

disp('meanError')
disp(mean(errors(:)))
