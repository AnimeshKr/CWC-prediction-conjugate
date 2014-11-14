function [nn, finalNN] = superVisedTrain_cwc_2layer_lib(sae1, sae2, X,y )
 
    no_features = size(sae1.Theta1,2)-1;
    %assert( nargin == 5 || nargin == 2, 'Number of arguments must be 5 or 2' );
    
%     if nargin == 5,
    
        %%%%%%%%%%%Reading input%%%%%%%%%%%%%%
        %[X,y] = data_extract();
%         X = X(1:end-1,:);
%         y = csvread('cwcDaily.csv');
%         y = y(5:end,:);
        assert(size(X,1) == size(y,1), 'dimensions not matching');
                        
%         X = X - repmat( initAvg, size(X,1), 1 );
%         X = X ./ repmat( initRan, size(X,1), 1 );
% %         X = X * U(:, (size(X,2) - no_features)+1:end );
%         X = X ./ repmat( ran, size(X,1), 1 );

%         fnights = csvread('fnightsFrom1871.csv');
%         fnights = fnights(5:end-1,:);
% 
%         assert(size(fnights,1) == size(y,1), 'dimensions of fnights are not matching');
%         save('traincwc_full.mat','X','y','fnights');

%     elseif nargin == 2,
%         
%         load traincwc_full.mat;
% 
%     end
    
%     requiredFnights = [11,12,13,14,15,16,17,18];
%     requiredPos = ismember( fnights, requiredFnights );
%     X = X(requiredPos,:);
%     y = y(requiredPos,:);
     
    
    %%%%%%%%%%Running feedForward%%%%%%%%%
    features1 = tanh_opt( [ ones(size(X,1),1) X ] * sae1.Theta1' );
    features2 = tanh_opt( [ ones(size(features1,1),1) features1 ] * sae2.Theta1' );
    
    
    %%%%%%%%%Dividing into validation set%%%%
    trainSize = floor(0.92*size(X,1));
    
    features2Val = features2( trainSize+1:end, : );
    features2 = features2( 1:trainSize, : );
    
    
    y = 100*y;
    meanValues = mean(y);
    yval = y( trainSize+1:end, : );
    y = y( 1:trainSize, : );
    
    %%%%%%%%%Setting options%%%%%%%%%%%
    input_layer_size = size(features2,2);
    hidden_layer_size = 100;
    num_labels = size(y,2);
    lambda= 0;
    initial_Theta1 = ( rand( hidden_layer_size, input_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+input_layer_size)) ;
    initial_Theta2 = ( rand( num_labels, hidden_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+num_labels)) ;
    initial_nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;

    costFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, features2, y, lambda);
    
    valcostFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, features2Val, yval, 0);
                               
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 150);
    [nn_params, cost] = fmincg(costFunc, initial_nn_params, options, valcostFunc, 0);

    % Obtain Theta1 and Theta2 back from nn_params
    nn.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    nn.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    [trainError, testError] = feedForward(nn,features2,y,features2Val,yval,meanValues);
    
    disp('trainError');
    disp(trainError);
    
    disp('testError');
    disp(testError);
    
    finalNN = fineTune_2layer_lib(nn, sae1, sae2, X, [y; yval]);
        
end
