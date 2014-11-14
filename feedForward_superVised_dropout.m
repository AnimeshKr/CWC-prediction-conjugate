%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function is used to feed forward the input over the NN archietecture
%% optimized at levels and finding out the error in prediction
%% parameters
%% nn: simple NN structure parameters( Input:hidden:output)
%% X: Training set
%% y: training output
%% Xval: Validation input
%% yVal: Validation output
%% mean Values: LPA

%% trainError, test Error : rms error for training and test set
%%  absTrainError, absTestError : absolute train and test errors
%% All errors are calculated for rainfall expressed as percent of LPA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function[ testError, a3val, yval ] = feedForward_superVised(nn,X,y,Xval,yval,meanValues,dropOutRatio)

    if ~exist('dropOutRatio','var')
        dropOutRatio = [0 0];
    end
    
%    m = size(X,1);
    %% layer 1( hidden layer activation : archieved from weight and tanH function)
%    X = X .* ( 1 - dropOutRatio(1) );
%    a2 = tanh_opt( [ ones(m,1) X ] * nn.Theta1' );
%    a2 = a2 .* ( 1 - dropOutRatio(2) );
%    a2 = [ ones(m,1) a2 ];
    %% layer2( output layer(linear layer)- output obtained from linear function))
%    a3 = a2 * nn.Theta2';
    
    %absTrainError = mean( mean( abs(a3 - y)* 100 ./ repmat(meanValues,size(y,1),1) ) );
%    trainError = rms( rms( (a3 - y) * 100 ./ repmat(meanValues,size(y,1),1) ) );
    %trainError=1;
    
    m = size(Xval,1);
    Xval = Xval .* ( 1 - dropOutRatio(1) );
    a2val = tanh_opt( [ ones(m,1) Xval ] * nn.Theta1' );
    a2val = a2val .* ( 1 - dropOutRatio(2) );
    a2val = [ones(m,1) a2val];
    a3val = a2val * nn.Theta2';
    
    %% errors for each test year is the output (averaged over 4 months of year)
    
    a3val = sum(a3val) * 100 ./ meanValues;
    yval = sum(yval) * 100 ./ meanValues;    
    testError = abs( a3val- yval );     

%    testError = rms( rms( (a3val - yval) * 100 ./ repmat(meanValues,size(yval,1),1) ) );

	

    %figure
    %plot(1:200,a3val(1:200,5),1:200,yval(1:200,5))
    %mean( abs(a3val - yval) * 100 ./ repmat(meanValues,size(yval,1),1))
end
