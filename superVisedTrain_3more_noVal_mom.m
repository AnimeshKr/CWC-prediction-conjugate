function [nn, finalNN] = superVisedTrain_3more_noVal_mom(sae1, sae2, sae3)
    
    load input_3more.mat;
    load output_south.mat;
    
    assert(size(X1,1) == size(y,1) && size(X2,1) == size(y,1) && size(X3,1) == size(y,1), 'dimensions not matching');
                        
    %%%%%%%%%%Running feedForward%%%%%%%%%
    m = size(X1,1);
    features1 = [ tanh_opt( [ ones(m,1) X1 ] * sae1.Theta1' )  tanh_opt( [ ones(m,1) X2 ] * sae2.Theta1' )  tanh_opt( [ ones(m,1) X3 ] * sae3.Theta1' )];
    mean1 = zeros(1,size(features1,2));
%     mean1 = mean( features1 );
    features1 = features1 - repmat(mean1, size(features1,1), 1);
    
    y = y*100;
    origfeatures = features1;
    meanValues = mean(y)
%     pause;
    %Settingup values
    input_layer_size = size(origfeatures,2);
    hidden_layer_size = 60;
    num_labels = size(y,2);
    lambda = 0.0;
    
    startYear = 2005;
    endYear = 2012;
    experiments = 1;
    dropOutRatios = [0 0.5];
        
    for i = 1:(endYear-startYear)+1,
        
        predictionYear = startYear + i - 1;
        errors(i,1) = predictionYear; 
        prevErrors(i,1) = predictionYear;
        
        %%%%Dividing into validation set, train set and test set%%%%
        X1test   = X1 (end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , :);
        X2test   = X2 (end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , :);
        X3test   = X3 (end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , :);
        featuresTest = origfeatures( end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , : );
        ytest = y( end - ( (2012-predictionYear)*122+121 ) : end - (2012-predictionYear)*122 , : );

        curYearX1   = X1 ( 1: end - ( (2012-predictionYear)*122+122 ) , :);
        curYearX2   = X2 ( 1: end - ( (2012-predictionYear)*122+122 ) , :);
        curYearX3   = X3 ( 1: end - ( (2012-predictionYear)*122+122 ) , :);
        curYearfeatures = origfeatures( 1:end - ( (2012-predictionYear)*122+122 ), : ); 
        curYearY = y( 1:end - ( (2012-predictionYear)*122+122 ), : );
        
        for exp = 1:experiments,
            
            disp( ['Predicting year ' num2str(predictionYear) ' experiment number ' num2str(exp)] );
   
%             pos = randperm( size(curYearfeatures,1) );
%             curYearfeatures = curYearfeatures(pos,:);  curYearY = curYearY(pos,:);
            
            costFunc = @(p,inp,out) costFunction_do(p, ...
                                               input_layer_size, ...
                                               hidden_layer_size, ...
                                               num_labels, inp, out, lambda, dropOutRatios);

            valcostFunc = @(p,inp,out) valcostFunction_do(p, ...
                                               input_layer_size, ...
                                               hidden_layer_size, ...
                                               num_labels, inp, out, dropOutRatios);

            initial_Theta1 = ( rand( hidden_layer_size, input_layer_size+1 ) - 0.5 ) * 1.2 * sqrt(6 / (hidden_layer_size+input_layer_size)) ;
            initial_Theta2 = ( rand( num_labels, hidden_layer_size+1 ) - 0.4 ) * 12 * sqrt(6 / (hidden_layer_size+num_labels)) ;
            nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;
            
            % Now, costFunction is a function that takes in only one argument (the
            % neural network parameters)
            options.batchSize = 500;
            options.epochs = 100;
            options.plot = 0;
            nn_params = fminmom_noLS(costFunc, valcostFunc, curYearfeatures, curYearY, featuresTest, ytest, nn_params, options,  1, 0);
                    
            % Obtain Theta1 and Theta2 back from nn_params
            nn.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                             hidden_layer_size, (input_layer_size + 1));

            nn.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                             num_labels, (hidden_layer_size + 1));

            nn.Theta1(1:5,1:8)
            a2 = tanh_opt([ones(size(featuresTest,1),1) featuresTest] * nn.Theta1');
            mean(a2)
            
            [trainError, prevErrors(i,2*exp), absTrainError, prevErrors(i,2*exp+1)] = feedForward_superVised_do(nn,curYearfeatures,curYearY,featuresTest,ytest,meanValues,dropOutRatios);

            disp('rms error');
            disp(prevErrors(i,2*exp));
            
            disp('abs error');
            disp(prevErrors(i,2*exp+1));
            
            disp('rms train error');
            disp(trainError);
            
            disp('abs Train error');
            disp(absTrainError);
            
            [finalNN, ~, errors(i, 3*exp-1), ~,errors(i, 3*exp), errors(i, 3*exp+1)] = fineTune_3more_noVal_mom(nn, sae1, sae2, sae3, curYearX1, curYearX2, curYearX3, curYearY, X1test, X2test, X3test, ytest, meanValues, mean1, predictionYear);
        end
    end
     
    disp('previous rms error');
    disp(mean(mean( prevErrors(:,2:2:end) )))
    
    disp('previous abs error');
    disp(mean(mean( prevErrors(:,3:2:end) )))
    
    disp('fine tuned rms error');
    disp(mean(mean(errors(:,2:3:3*exp+1))))
    
    disp('fine tuned abs error');
    disp(mean(mean(errors(:,3:3:3*exp+1))))
    
    disp('fine tuned mean min error');
    disp(mean(mean(errors(:,4:3:3*exp+1))))
    
    csvwrite( ['3autowithoutvalidatedEliminationErrorsfrom' num2str(startYear) 'to' num2str(endYear) 'hid' num2str(hidden_layer_size) 'lambda' num2str(lambda) '.csv'], errors );
    csvwrite( ['3autowithoutvalidatedprevEliminationErrorsfrom' num2str(startYear) 'to' num2str(endYear) 'hid' num2str(hidden_layer_size) 'lambda' num2str(lambda) '.csv'], prevErrors );

end
