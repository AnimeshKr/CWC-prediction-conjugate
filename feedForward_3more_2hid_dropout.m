%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function is used to feed forward the input over the NN archietecture
%% optimized at levels and finding out the error in prediction
%% parameters
%% nn: Final NN structure parameters( Input:hidden:output) (after fine tunning)
%% X: Training set
%% y: training output
%% Xval: Validation input
%% yVal: Validation output
%% mean Values: LPA
%% trainError, test Error : absolute error for training and test set
%% a4: Predicted rainfall values for test set
%% yval: Actual rainfall values for test set
%% All errors are calculated for rainfall expressed as percent of LPA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[testError, a4, yval] = feedForward_3more_2hid(nn,X1,X2,X3,y,X1val,X2val,X3val,yval,meanValues,mean1,dropOutRatios)

    if ~exist('dropOutRatios','var')
        dropOutRatios = [0 0];
    end
    
    m = size(X1,1);
    a11 = [ ones(m,1) X1 ];
    a12 = [ ones(m,1) X2 ];
    a13 = [ ones(m,1) X3 ];
    a21 = tanh_opt( a11 * nn.Theta11' );
    a22 = tanh_opt( a12 * nn.Theta12' );
    a23 = tanh_opt( a13 * nn.Theta13' ); 
    a2 = [a21 a22 a23];
    a2 = a2 .* ( 1 - dropOutRatios(1) );
    a2 = [ ones(m,1) a2 ];
    a3 = tanh_opt( (a2 - repmat( [0 mean1], m, 1 )) * nn.Theta2' );
    a3 = a3 .* ( 1 - dropOutRatios(2) );
    a3 = [ ones(m,1) a3 ];
    a4 = a3 * nn.Theta3';
    
%     disp('Mean activations after final feed forward')
%     std(a2)
%     std(a3)
%     mean(a2)
%     mean(a3)
%     trainError = mean( mean( abs(a4 - y)* 100 ./ repmat(meanValues,size(y,1),1) ) );
%     trainError = rms( rms( abs(a4 - y) * 100 ./ repmat(meanValues,size(y,1),1) ) ); 
   
    
    
    m = size(X1val,1);
    a11 = [ ones(m,1) X1val ];
    a12 = [ ones(m,1) X2val ];
    a13 = [ ones(m,1) X3val ];
    a21 = tanh_opt( a11 * nn.Theta11' );
    a22 = tanh_opt( a12 * nn.Theta12' );
    a23 = tanh_opt( a13 * nn.Theta13' ); 
    a2 = [a21 a22 a23];
    a2 = a2 .* ( 1 - dropOutRatios(1) );
    a2 = [ ones(m,1) a2 ];
    a3 = tanh_opt( (a2 - repmat( [0 mean1], m, 1 )) * nn.Theta2' );
    a3 = a3 .* ( 1 - dropOutRatios(2) );
    a3 = [ ones(m,1) a3 ];
    a4 = a3 * nn.Theta3';
    
%     plot( 1:size(a4,1), a4, 1:size(a4,1),  yval );

    a4 = sum(a4) * 100 / meanValues;
    yval = sum(yval) * 100 / meanValues;
    testError = abs( a4 - yval );
    
%     testError = rms( rms( abs(a4 - yval) * 100 ./ repmat(meanValues,size(yval,1),1) ) );


 end
