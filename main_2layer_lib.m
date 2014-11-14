clear all;
clc;

new = 0;

%if new == 1,
    [X,y] = data_extract();
    

    %%%%normalizing for pca%%%%
    initAvg = mean(X);
    X = X - repmat(initAvg, size(X,1), 1);
    initRan = sqrt( mean(X.^2) );
    X = X ./ repmat(initRan, size(X,1), 1);
    [X, U, error, ~] = pca(X,0.9999);

    %%%Normalizing after pca%%%%
    ran1 = sqrt( var(X) );
    X = X ./ repmat(ran1, size(X,1), 1);

%     save( 'sae.mat', 'U', 'initAvg', 'initRan', 'pos', 'ran1', 'X' );
% else
%     load sae.mat;
% end
X_init=X;
y_init=y;


trainSize = floor(0.92*size(X,1));
Xval = X(trainSize+1:end,:);
X = X(1:trainSize,:);
pos = randperm(size(X,1));
X = X(pos,:); 
input_layer_size = size(X,2);
hidden_layer_size = 400;
num_labels = size(X,2);
lambda= 0;
initial_Theta1 = ( rand( hidden_layer_size, input_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+input_layer_size)) ;
initial_Theta2 = ( rand( num_labels, hidden_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+num_labels)) ;
nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;
noiseFraction = 0.2;


for i = 1:10,
    
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
    options = optimset('MaxIter', 6);
    [nn_params, cost] = fmincg(costFunc, nn_params, options, valcostFunc,0);

end
    
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


sae1.Theta1 = Theta1;
sae1.Theta2 = Theta2;
             
[trainError, testError] = feedForward( sae1,X,X,Xval,Xval );
disp('trainError');
disp(trainError);
disp('testError');
disp(testError);

save( ['autoEncoderParameters1Layer' num2str(hidden_layer_size) '.mat'], 'sae1');






%%%Second layer of autoencoder%%%%
features1 = tanh_opt( [ones(size(X,1),1) X] * Theta1' );
features1Val = tanh_opt( [ones(size(Xval,1),1) Xval] * Theta1' );
input_layer_size = size(features1,2);
hidden_layer_size = 250;
num_labels = size(features1,2);
lambda= 0;
initial_Theta1 = ( rand( hidden_layer_size, input_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+input_layer_size)) ;
initial_Theta2 = ( rand( num_labels, hidden_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+num_labels)) ;
nn2_params =[ initial_Theta1(:); initial_Theta2(:) ] ;
noiseFraction = 0.2;

for i = 1:6,
    
    %Add noise
    randpos = rand(size(features1)) < noiseFraction;
    batch_x = features1;
    batch_x(randpos) = randi(3,sum(randpos(:)),1)-2;
    
    costFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, batch_x, features1, lambda);

    valcostFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, features1Val, features1Val, 0);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 7);
    [nn2_params, cost] = fmincg(costFunc, nn2_params, options, valcostFunc,0);

end
    
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn2_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn2_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


sae2.Theta1 = Theta1;
sae2.Theta2 = Theta2;
             
%save( ['autoEncoderParameters2Layer' num2str(hidden_layer_size) '.mat'], 'sae2');






%if new == 1,
    [nn, finalNN] = superVisedTrain_cwc_2layer_lib( sae1, sae2, X_init,y_init);
%else
   % [nn, finalNN] = superVisedTrain_cwc_2layer_lib( sae1, sae2 );
%end
