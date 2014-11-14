function[trainError, testError] = feedForward(nn,X,y,Xval,yval,meanValues)

    m = size(X,1);
    a2 = [ ones(m,1) tanh_opt( [ ones(m,1) X ] * nn.Theta1' ) ];
    a3 = a2 * nn.Theta2';
    
    if nargin == 6,
        trainError = mean( mean( abs(a3 - y)* 100 ./ repmat(meanValues,size(y,1),1) ) );
    else
        trainError = mean( mean( abs(a3 - y) ) );
    end
    
    m = size(Xval,1);
    a2val = [ ones(m,1) tanh_opt( [ ones(m,1) Xval ] * nn.Theta1' ) ];
    a3val = a2val * nn.Theta2';
    
    if nargin == 6,
        testError = mean( mean( abs(a3val - yval) * 100 ./ repmat(meanValues,size(yval,1),1) ) );
    else
        testError = mean( mean( abs(a3val - yval) ) );
    end
    %figure
    %plot(1:200,a3val(1:200,5),1:200,yval(1:200,5))
    %mean( abs(a3val - yval) * 100 ./ repmat(meanValues,size(yval,1),1))
end