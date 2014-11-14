function[trainError, testError] = feedForward_2hid(nn,X,y,Xval,yval,meanValues)

    m = size(X,1);
    a1 = [ ones(m,1) X ];
    a2 = [ ones(m,1) tanh_opt( a1 * nn.Theta1' ) ];
    a3 = [ ones(m,1) tanh_opt( a2 * nn.Theta2' ) ];
    a4 = a3 * nn.Theta3';
    
    if nargin == 6,
        trainError = mean( mean( abs(a4 - y)* 100 ./ repmat(meanValues,size(y,1),1) ) );
    else
        trainError = mean( mean( abs(a4 - y) ) );
    end
    
    m = size(Xval,1);
    a1 = [ ones(m,1) Xval ];
    a2 = [ ones(m,1) tanh_opt( a1 * nn.Theta1' ) ];
    a3 = [ ones(m,1) tanh_opt( a2 * nn.Theta2' ) ];
    a4 = a3 * nn.Theta3';
    
    if nargin == 6,
        testError = mean( mean( abs(a4 - yval) * 100 ./ repmat(meanValues,size(yval,1),1) ) );
    else
        testError = mean( mean( abs(a4 - yval) ) );
    end
    
    Error =( mean( abs(a4 - yval) * 100 ./ repmat(meanValues,size(yval,1),1) ) );
    disp(Error);
    te=mean(Error(1,50:end));
    disp('te='); disp(te);
    er=min(Error);
%     Error=reshape(Error,14,15);
%     
%     imagesc(Error);
%     figure
%     plot(1:200,a4(1:200,172),1:200,yval(1:200,172))
%     figure
%     plot(1:200,a4(1:200,158),1:200,yval(1:200,158))
    disp('min Error'); disp(er);
    
    
end
