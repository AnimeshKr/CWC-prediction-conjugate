function [finalNN] = fineTune_lib( nn, sae, X, y )
    
%     

    trainSize = ceil( 0.85 * size(X,1) );
    Xval = X( trainSize+1:end, : );
    yval = y( trainSize+1:end, : );
    pos = randperm( size(Xval, 1) );
    Xval = Xval(pos,:);   yval = yval(pos,:);
    X = X( 1:trainSize, : );
    y = y( 1:trainSize, : );
    X = [X;X(13000:end,:)];
    y = [y;y(13000:end,:)];
    pos = randperm( size(X, 1) );
    X = X(pos,:);   y = y(pos,:);
    
    %%%%%%%%%Setting options%%%%%%%%%%%
    input_layer_size = size( sae.Theta1, 2 )-1;
    hidden_layer1_size = size( sae.Theta1, 1 );
    hidden_layer2_size = size( nn.Theta1, 1 );
    num_labels = size(y,2);
    lambda= 0.05;
    initial_Theta1 = sae.Theta1;
    initial_Theta2 = nn.Theta1;
    initial_Theta3 = nn.Theta2;
    initial_nn_params =[ initial_Theta1(:); initial_Theta2(:); initial_Theta3(:) ] ;

    costFunc = @(p) costFunction_2hid(p, ...
                                       input_layer_size, ...
                                       hidden_layer1_size, ...
                                       hidden_layer2_size, ...
                                       num_labels, X, y, lambda);
    
    valcostFunc = @(p) costFunction_2hid(p, ...
                                       input_layer_size, ...
                                       hidden_layer1_size, ...
                                       hidden_layer2_size, ...
                                       num_labels, Xval, yval, 1);
                                   
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 200);
    [nn_params, cost] = fmincg(costFunc, initial_nn_params, options, valcostFunc,0);

    % Obtain Theta1 and Theta2 back from nn_params
    finalNN.Theta1 = reshape( nn_params(1:numel(initial_Theta1)), size(initial_Theta1) );
    finalNN.Theta2 = reshape( nn_params( numel(initial_Theta1)+1:numel(initial_Theta1)+numel(initial_Theta2) ) , size(initial_Theta2) );
    finalNN.Theta3 = reshape( nn_params( end - ( numel(initial_Theta3)-1 ) : end ), size(initial_Theta3) );
    
    [trainError, testError] = feedForward_2hid( finalNN, X, y, Xval, yval, mean([y;yval]) );
    disp('trainError')
    trainError
    disp('testError')
    testError
    
    
end
