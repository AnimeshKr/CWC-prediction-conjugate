clear all;
clc;

%%Loading and preprocessing data
load input_noCov.mat;
disp('loaded input');
randpos = randperm(size(X,2));
X = X(:,randpos);
initAvg = mean(X);
X = X - repmat(initAvg, size(X,1), 1);
initRan = sqrt( mean(X.^2) );
X = X ./ repmat(initRan, size(X,1), 1);
[X, U, error, ~] = pca(X,0.999);
ran1 = 2*std(X);
X = X ./ repmat(ran1, size(X,1), 1);

firstIn = floor(size(X,2)/3);
secondIn = floor(2*size(X,2)/3);
X1 = X(:,1:firstIn);
X2 = X(:,firstIn+1:secondIn);
X3 = X(:,secondIn+1:end);

save('input_3more.mat','X1','X2','X3');

randpos = randperm(size(X,1));
X1 = X1(randpos,:);
X2 = X2(randpos,:);
X3 = X3(randpos,:);

trainSize = floor(0.92*size(X,1));

X1val = X1(trainSize+1:end,:);
X1 = X1(1:trainSize,:);

X2val = X2(trainSize+1:end,:);
X2 = X2(1:trainSize,:);

X3val = X3(trainSize+1:end,:);
X3 = X3(1:trainSize,:);

dropOutRatios = [0 0];
input_layer_size = size(X1,2);
hidden_layer_size = floor( 0.4*input_layer_size );
num_labels = size(X1,2);
lambda = 0.02;
initial_Theta1 = [ zeros(hidden_layer_size,1) ( rand( hidden_layer_size, input_layer_size ) - 0.5 ) * 1.2 * sqrt(6 / (hidden_layer_size+input_layer_size)) ] ;
initial_Theta2 = [ zeros(num_labels,1) ( rand( num_labels, hidden_layer_size ) - 0.5 ) * 6 * sqrt(6 / (hidden_layer_size+num_labels)) ] ;
nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;
noiseFraction = 0.3;

for i = 1:8,
    
    %Add noise
    randpos = rand(size(X1)) < noiseFraction;
    batch_x = X1;

    batch_x(randpos) = randi(3,sum(randpos(:)),1)-2;
    
%     randomBits = randi(2,sum(randpos(:)),1)-1;
%     randomBits(randomBits==0) = -1;
%     batch_x(randpos) = randomBits;
    
%     batch_x(randpos) = 0;

    costFunc = @(p) costFunction_do(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, batch_x, X1, lambda, dropOutRatios);

    valcostFunc = @(p) valcostFunction_do(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X1val, X1val, dropOutRatios);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 15);
    [nn_params, cost] = fmincg(costFunc, nn_params, options, valcostFunc,0,1);

end
    
% Obtain Theta1 and Theta2 back from nn_params
sae1.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

sae1.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
         
[trainError, testError] = feedForward( sae1,X1,X1,X1val,X1val );
disp('trainError')
disp(trainError);
disp('testError');
disp(testError);

% save( ['autoEncoder1Parameters' num2str(hidden_layer_size) '.mat'], 'sae1');


% pause;






input_layer_size = size(X2,2);
hidden_layer_size = floor( 0.4*input_layer_size );
num_labels = size(X2,2);
lambda = 0.02;
initial_Theta1 = [ zeros(hidden_layer_size,1) ( rand( hidden_layer_size, input_layer_size ) - 0.5 ) * 1.2 * sqrt(6 / (hidden_layer_size+input_layer_size)) ] ;
initial_Theta2 = [ zeros(num_labels,1) ( rand( num_labels, hidden_layer_size ) - 0.5 ) * 6 * sqrt(6 / (hidden_layer_size+num_labels)) ] ;
nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;
noiseFraction = 0.3;
 
for i = 1:8,
    
    %Add noise
    randpos = rand(size(X2)) < noiseFraction;
    batch_x = X2;

    batch_x(randpos) = randi(3,sum(randpos(:)),1)-2;
    
%     randomBits = randi(2,sum(randpos(:)),1)-1;
%     randomBits(randomBits==0) = -1;
%     batch_x(randpos) = randomBits;
    
%     batch_x(randpos) = 0;

    costFunc = @(p) costFunction_do(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, batch_x, X2, lambda, dropOutRatios);

    valcostFunc = @(p) valcostFunction_do(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X2val, X2val, dropOutRatios);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 15);
    [nn_params, cost] = fmincg(costFunc, nn_params, options, valcostFunc,0,1);

end
    
% Obtain Theta1 and Theta2 back from nn_params
sae2.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

sae2.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
         
[trainError, testError] = feedForward( sae2,X2,X2,X2val,X2val );
disp('trainError')
disp(trainError);
disp('testError');
disp(testError);

% save( ['autoEncoder2Parameters' num2str(hidden_layer_size) '.mat'], 'sae2');

% pause;







input_layer_size = size(X3,2);
hidden_layer_size = floor( 0.4*input_layer_size );
num_labels = size(X3,2);
lambda = 0.02;
initial_Theta1 = [ zeros(hidden_layer_size,1) ( rand( hidden_layer_size, input_layer_size ) - 0.5 ) * 1.2 * sqrt(6 / (hidden_layer_size+input_layer_size)) ] ;
initial_Theta2 = [ zeros(num_labels,1) ( rand( num_labels, hidden_layer_size ) - 0.5 ) * 6 * sqrt(6 / (hidden_layer_size+num_labels)) ] ;
nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;
noiseFraction = 0.3;

for i = 1:8,
    
    %Add noise
    randpos = rand(size(X3)) < noiseFraction;
    batch_x = X3;

    batch_x(randpos) = randi(3,sum(randpos(:)),1)-2;
    
%     randomBits = randi(2,sum(randpos(:)),1)-1;
%     randomBits(randomBits==0) = -1;
%     batch_x(randpos) = randomBits;
    
%     batch_x(randpos) = 0;

    costFunc = @(p) costFunction_do(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, batch_x, X3, lambda, dropOutRatios);

    valcostFunc = @(p) valcostFunction_do(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X3val, X3val, dropOutRatios);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 15);
    [nn_params, cost] = fmincg(costFunc, nn_params, options, valcostFunc,0,1);

end
    
% Obtain Theta1 and Theta2 back from nn_params
sae3.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

sae3.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
         
[trainError, testError] = feedForward( sae3,X3,X3,X3val,X3val );
disp('trainError')
disp(trainError);
disp('testError');
disp(testError);

% save( ['autoEncoder3Parameters' num2str(hidden_layer_size) '.mat'], 'sae3');
% pause;

[nn, finalNN] = superVisedTrain_3more_noVal_lib( sae1, sae2, sae3 );
