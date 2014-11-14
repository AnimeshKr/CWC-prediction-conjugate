function [nn, finalNN] = superVisedTrain_cwc_libAK(sae, X,y )
   
    X_init=X;
    y_init=y;
    %%%%%%%%%%Running feedForward%%%%%%%%%
    features1 = tanh_opt( [ ones(size(X,1),1) X ] * sae.Theta1' );
    
    
    %%%%%%%%%Dividing into validation set%%%%
    trainSize = floor(0.85*size(X,1));
    
    features1Val = features1( trainSize+1:end, : );
    features1 = features1( 1:trainSize, : );
    features1 = [features1;features1(13000:end,:)];    
    y = y*100;
    meanValues = mean(y);
    yval = y( trainSize+1:end, : );
    y = y( 1:trainSize, : );
    y=[y;y(13000:end,:)];
    
    %%%%%%%%%Setting options%%%%%%%%%%%
    input_layer_size = size(features1,2);
    hidden_layer_size = 200;
    num_labels = size(y,2);
    lambda= 0.05;
    initial_Theta1 = ( rand( hidden_layer_size, input_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+input_layer_size)) ;
    initial_Theta2 = ( rand( num_labels, hidden_layer_size+1 ) - 0.5 ) * 5 * sqrt(6 / (hidden_layer_size+num_labels)) ;
    initial_nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;

    costFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, features1, y, lambda);
    
    valcostFunc = @(p) costFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, features1Val, yval, 0);
                               
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 120);
    [nn_params, cost] = fmincg(costFunc, initial_nn_params, options, valcostFunc, 0);

    % Obtain Theta1 and Theta2 back from nn_params
    nn.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    nn.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    [trainError, testError] = feedForward(nn,features1,y,features1Val,yval,meanValues);
    
    disp('trainError');
    disp(trainError);
    
    disp('testError');
    disp(testError);
    
    finalNN = fineTune_lib(nn, sae, X_init, y_init*100);
        
end
