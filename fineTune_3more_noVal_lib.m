function [finalNN, trainError, testError, absTrainError, absTestError, minTestError] = fineTune_3more_noVal_lib( nn, sae1, sae2, sae3, origX1, origX2, origX3, origY, X1test, X2test, X3test, ytest, meanValues, mean1, predictionYear)
 
    input_layer1_size = size( sae1.Theta1, 2 )-1;
    input_layer2_size = size( sae2.Theta1, 2 )-1;
    input_layer3_size = size( sae3.Theta1, 2 )-1;
    hidden_layer11_size = size( sae1.Theta1, 1 );
    hidden_layer12_size = size( sae2.Theta1, 1 );
    hidden_layer13_size = size( sae3.Theta1, 1 );
    hidden_layer2_size = size( nn.Theta1, 1 );
    num_labels = size(origY,2);
    lambda = 0.0;
    initial_Theta11 = sae1.Theta1;
    initial_Theta12 = sae2.Theta1;
    initial_Theta13 = sae3.Theta1;
    initial_Theta2 = nn.Theta1;
    initial_Theta3 = nn.Theta2;
    initial_nn_params =[ initial_Theta11(:); initial_Theta12(:); initial_Theta13(:); initial_Theta2(:); initial_Theta3(:) ] ;
    dropOutRatios = [0 0];
%     pos = randperm( size(origX1, 1) );
%     origX1 = origX1(pos,:);   origX2 = origX2(pos,:);  origX3 = origX3(pos,:);  origY = origY(pos,:);
    
    costFunc = @(p) costFunction_3more_2hid_do(p, ...
                                       input_layer1_size, ...
                                       input_layer2_size, ...
                                       input_layer3_size, ...
                                       hidden_layer11_size, ...
                                       hidden_layer12_size, ...
                                       hidden_layer13_size, ...
                                       hidden_layer2_size, ...
                                       num_labels, origX1, origX2, origX3, origY, lambda, mean1, dropOutRatios);

    valcostFunc = @(p) valcostFunction_3more_2hid_do(p, ...
                                       input_layer1_size, ...
                                       input_layer2_size, ...
                                       input_layer3_size, ...
                                       hidden_layer11_size, ...
                                       hidden_layer12_size, ...
                                       hidden_layer13_size, ...
                                       hidden_layer2_size, ...
                                       num_labels, X1test, X2test, X3test, ytest, mean1, dropOutRatios);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    options = optimset('MaxIter', 150);
    [nn_params, ~, ~, ~] = fmincg(costFunc, initial_nn_params, options, valcostFunc, 1);

    % Obtain Theta1 and Theta2 back from nn_params
    finalNN.Theta11 = reshape( nn_params( 1:numel(initial_Theta11) ), size(initial_Theta11) );
    finalNN.Theta12 = reshape( nn_params( numel(initial_Theta11)+1 : numel(initial_Theta11)+numel(initial_Theta12) ), size(initial_Theta12) );
    finalNN.Theta13 = reshape( nn_params( numel(initial_Theta11)+numel(initial_Theta12)+1 : numel(initial_Theta11)+numel(initial_Theta12)+numel(initial_Theta13) ), size(initial_Theta13) );
    finalNN.Theta2 = reshape( nn_params( numel(initial_Theta11)+numel(initial_Theta12)+numel(initial_Theta13)+1 : numel(initial_Theta11)+numel(initial_Theta12)+numel(initial_Theta13)+numel(initial_Theta2) ) , size(initial_Theta2) );
    finalNN.Theta3 = reshape( nn_params( end - ( numel(initial_Theta3)-1 ) : end ), size(initial_Theta3) );

%     [trainError, testError, ~, ~, minTestError, absTrainError, absTestError] = feedForward_3more_2hid_do( finalNN, origX1, origX2, origX3, origY, X1test, X2test, X3test, ytest, meanValues, mean1, dropOutRatios );
    [absTrainError, absTestError, minTestError] = feedForward_3more_2hid_do( finalNN, origX1, origX2, origX3, origY, X1test, X2test, X3test, ytest, meanValues, mean1, dropOutRatios );
    trainError = 1;
    testError = 1;
    disp('trainError')
    disp(trainError)
    disp('testError')
    disp(testError)
    disp('minTestError')
    disp(minTestError)
    disp('absTrainError')
    disp(absTrainError)
    disp('absTestError')
    disp(absTestError)
end
