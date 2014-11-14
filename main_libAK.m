clear all;
clc;

addpath('d:\Climate DM project\regression cwc\');
% [X,y] = data_extract();
load input_cov.mat;
load output_south.mat;
  
%%%%normalizing for pca%%%%
initAvg = mean(X);
X = X - repmat(initAvg, size(X,1), 1);
initRan = sqrt( mean(X.^2) );
X = X ./ repmat(initRan, size(X,1), 1);
[X, U, error, ~] = pca(X,0.9999);

%%%Normalizing after pca%%%%
 ran1 = 3.25*sqrt( var(X) );
X = X ./ repmat(ran1, size(X,1), 1);
X_init=X;
y_init=y;

trainSize = floor(0.85*size(X,1));
Xval = X(trainSize+1:end,:);
X = X(1:trainSize,:);
X=[X;X(13000:end,:)];
pos = randperm(size(X,1));
X = X(pos,:); 

input_layer_size = size(X,2);
hidden_layer_size = 400;
num_labels = size(X,2);
lambda= 0.05;
initial_Theta1 = ( rand( hidden_layer_size, input_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+input_layer_size)) ;
initial_Theta2 = ( rand( size(X,2), hidden_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+size(X,2))) ;
nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;
noiseFraction = 0.2;

for i = 1:4,
    
    %Add noise
    randpos = rand(size(X)) < noiseFraction;
    batch_x = X;
    batch_x(randpos) = randi(3,sum(randpos(:)),1)-2;
    
    costFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, batch_x, X, lambda);

    valcostFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, Xval, Xval, 0);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 15);
    [nn_params, cost] = fmincg(costFunc, nn_params, options, valcostFunc,0);

end
    
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


sae.Theta1 = Theta1;
sae.Theta2 = Theta2;
             
[trainError, testError] = feedForward( sae,X,X,Xval,Xval );
disp('trainError')
disp(trainError);
disp('testError');
disp(testError);

% save( ['autoEncoderParameters' num2str(hidden_layer_size) '.mat'], 'Theta1', 'Theta2');


% if new == 1,
%     [nn, finalNN] = superVisedTrain_cwc_lib( sae, U, initRan, initAvg, ran1 );
% else
[nn, finalNN] = superVisedTrain_cwc_libAK( sae,X_init,y_init );
%end

