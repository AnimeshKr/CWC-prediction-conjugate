function[trainError, testError, a4, yval, minTestError, absTrainError, absTestError] = feedForward_3more_2hid(nn,X1,X2,X3,y,X1val,X2val,X3val,yval,meanValues,mean1,predictionYear)

    m = size(X1,1);
    a11 = [ ones(m,1) X1 ];
    a12 = [ ones(m,1) X2 ];
    a13 = [ ones(m,1) X3 ];
    a21 = [ ones(m,1) tanh_opt( a11 * nn.Theta11' ) ];
    a22 = tanh_opt( a12 * nn.Theta12' );
    a23 = tanh_opt( a13 * nn.Theta13' ); 
    a2 = [a21 a22 a23];
    a3 = [ ones(m,1) tanh_opt( (a2 - repmat( [0 mean1], m, 1 )) * nn.Theta2' ) ];
    a4 = a3 * nn.Theta3';
    
    absTrainError = mean( mean( abs(a4 - y)* 100 ./ repmat(meanValues,size(y,1),1) ) );
    trainError = rms( rms( (a4 - y) * 100 ./ repmat(meanValues,size(y,1),1) ) ); 
   
    
    m = size(X1val,1);
    a11 = [ ones(m,1) X1val ];
    a12 = [ ones(m,1) X2val ];
    a13 = [ ones(m,1) X3val ];
    a21 = [ ones(m,1) tanh_opt( a11 * nn.Theta11' ) ];
    a22 = tanh_opt( a12 * nn.Theta12' );
    a23 = tanh_opt( a13 * nn.Theta13' );
    a2 = [a21 a22 a23];
    a3 = [ ones(m,1) tanh_opt( (a2 - repmat( [0 mean1], m, 1 )) * nn.Theta2' ) ];
    a4 = a3 * nn.Theta3';
    
    absTestError = mean( abs(a4 - yval) * 100 ./ repmat(meanValues,size(yval,1),1) );
    testError = rms( abs(a4 - yval) * 100 ./ repmat(meanValues,size(yval,1),1) );
    disp('absolute test error');
    absTestError
    minTestError = min( absTestError );
    absTestError = mean( absTestError );
    testError = rms( testError );
    
    figure
    plot( 1:size(a4,1), a4(:,6), 1:size(a4,1), yval(:,6) );
    xlabel('time');
    ylabel('Cloud water content');
    legend('predicted','desired');
    title(['Predicted vs desired output for ' num2str(predictionYear) ' year']);
    
end