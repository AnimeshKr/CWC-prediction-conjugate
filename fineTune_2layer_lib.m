function [finalNN] = fineTune_2layer_lib( nn, sae1, sae2, X, y )
    
    
    trainSize = ceil( 0.92 * size(X,1) );
    Xval = X( trainSize+1:end, : );
    yval = y( trainSize+1:end, : );
    
    X = X( 1:trainSize, : );
    y = y( 1:trainSize, : );
    pos = randperm( size(X, 1) );
    X = X(pos,:);   y = y(pos,:);

    
    %%%%%%%%%Setting options%%%%%%%%%%%
    input_layer_size = size( sae1.Theta1, 2 )-1;
    hidden_layer1_size = size( sae1.Theta1, 1 );
    hidden_layer2_size = size( sae2.Theta1, 1 );
    hidden_layer3_size = size( nn.Theta1, 1 );
    num_labels = size(y,2);
    lambda = 0.05;
    initial_Theta1 = sae1.Theta1;
    initial_Theta2 = sae2.Theta1;
    initial_Theta3 = nn.Theta1;
    initial_Theta4 = nn.Theta2;
    initial_nn_params =[ initial_Theta1(:); initial_Theta2(:); initial_Theta3(:); initial_Theta4(:) ] ;

    costFunc = @(p) costFunction_3hid(p, ...
                                       input_layer_size, ...
                                       hidden_layer1_size, ...
                                       hidden_layer2_size, ...
                                       hidden_layer3_size, ...
                                       num_labels, X, y, lambda);
    
    valcostFunc = @(p) costFunction_3hid(p, ...
                                       input_layer_size, ...
                                       hidden_layer1_size, ...
                                       hidden_layer2_size, ...
                                       hidden_layer3_size, ...
                                       num_labels, Xval, yval, 0);
                                   
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 250);
    [nn_params, cost] = fmincg(costFunc, initial_nn_params, options, valcostFunc,0);

    % Obtain Theta1 and Theta2 back from nn_params
    finalNN.Theta1 = reshape( nn_params( 1:numel(initial_Theta1) ), size(initial_Theta1) );
    finalNN.Theta2 = reshape( nn_params( numel(initial_Theta1)+1:numel(initial_Theta1)+numel(initial_Theta2) ) , size(initial_Theta2) );
    finalNN.Theta3 = reshape( nn_params( numel(initial_Theta1)+numel(initial_Theta2)+1 : end - numel(initial_Theta4) ), size(initial_Theta3) );
    finalNN.Theta4 = reshape( nn_params( end - ( numel(initial_Theta4)-1 ) : end ), size(initial_Theta4) );
    
    [trainError, testError] = feedForward_3hid( finalNN, X, y, Xval, yval, mean([y;yval]) );
    disp('trainError')
    trainError
    disp('testError')
    testError
    
end