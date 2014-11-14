function sae = train_1part_AE(X,Xtest,lambda,noiseFraction,hidtoinputRatio, initial_Theta1,initial_Theta2)

    disp('      Entered Auto encoder');
    input_layer_size = size(X,2);
    hidden_layer_size = floor( hidtoinputRatio * input_layer_size );
    num_labels = size(X,2);
    weightsGiven = 1;
    if ~exist('initial_Theta1','var') || ~exist('initial_Theta2','var'),
        weightsGiven = 0;
        initial_Theta1 = [ zeros(hidden_layer_size,1) ( rand( hidden_layer_size, input_layer_size ) - 0.5 ) * 1.2 * sqrt(6 / (hidden_layer_size+input_layer_size)) ];
        initial_Theta2 = [ zeros(num_labels,1) ( rand( num_labels, hidden_layer_size ) - 0.5 ) * 6 * sqrt(6 / (hidden_layer_size+num_labels)) ];
    end
    nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;

    
    
    
sae1.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

sae1.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
            mean(X)
             mean( tanh_opt( [ ones(size(X,1),1) X ] * sae1.Theta1' ) )

%     pause;
    
    
    for i = 1:10,

        %Add noise
        randpos = rand(size(X)) < noiseFraction;
        batch_x = X;

        batch_x(randpos) = randi(3,sum(randpos(:)),1)-2;

    %     randomBits = randi(2,sum(randpos(:)),1)-1;
    %     randomBits(randomBits==0) = -1;
    %     batch_x(randpos) = randomBits;

    %     batch_x(randpos) = 0;

        costFunc = @(p) costFunction(p, ...
                                           input_layer_size, ...
                                           hidden_layer_size, ...
                                           num_labels, batch_x, X, lambda);

        valcostFunc = @(p) costFunction(p, ...
                                           input_layer_size, ...
                                           hidden_layer_size, ...
                                           num_labels, Xtest, Xtest, 0);

        % Now, costFunction is a function that takes in only one argument (the
        % neural network parameters)
        if weightsGiven
            options = optimset('MaxIter',5);
        else
            options = optimset('MaxIter', 15);
        end
        [nn_params, cost] = fmincg(costFunc, nn_params, options, valcostFunc,0,1);

    end

    % Obtain Theta1 and Theta2 back from nn_params
    sae.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    sae.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

%     [trainError, testError] = feedForward( sae,X,X,Xtest,Xtest );
%     disp('trainError');
%     disp(trainError);
%     disp('testError');
%     disp(testError);
    disp('      Leaving Auto Encoder');
end