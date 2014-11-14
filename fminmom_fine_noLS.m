function [w, opt_epoch] = fminmom_fine_noLS(f, fval, X1, X2, X3, y, X1test, X2test, X3test, ytest, w, options,  best, display, normalize)
    
    velocity = zeros(size(w));
    alpha = 0.001;
    momentum_i = 0.5;
    momentum_f = 0.9;
    num_batches = floor( size(X1,1) / options.batchSize );
    minTestCost = Inf;
    batchSize = options.batchSize;
    wOpt = w;
        
    opt_epoch = 0;
    if isfield(options,'plot') && options.plot == 1, fhandle = figure(); end
    for epoch = 1:options.epochs,
        
        pos = randperm(size(X1,1));
        X1 = X1(pos,:);  X2 = X2(pos,:);    X3 = X3(pos,:);    y = y(pos,:);
        
        if epoch < options.epochs / 2
                momentum = momentum_i + (momentum_f - momentum_i)*epoch*2/(options.epochs);
        else momentum = momentum_f;
        end
        
        for batch = 1:num_batches,
            
            curX1 = X1( (batch-1)*batchSize+1 : batch*batchSize, : );
            curX2 = X2( (batch-1)*batchSize+1 : batch*batchSize, : );
            curX3 = X3( (batch-1)*batchSize+1 : batch*batchSize, : );
            curY = y( (batch-1)*batchSize+1 : batch*batchSize, : );
            
            [trainCost, grad] = f( w, curX1, curX2, curX3, curY );
%             velocity = velocity * momentum - (1 - momentum)*alpha*grad;
            velocity = velocity * momentum - alpha * grad;
            w = w + velocity;
%               w = w - alpha * grad;            
        end
        trainCst(epoch) = f( w, X1, X2, X3, y );
        testCst(epoch) = fval( w, X1test, X2test, X3test, ytest ); 
        testCost = testCst(epoch);
        
        if best && testCost < minTestCost,    minTestCost = testCost;    opt_epoch = epoch;      wOpt = w;   end
        if display, 
            disp( ['epoch ' num2str(epoch) ' : trainCost:' num2str(trainCost) '  ' num2str(trainCst(epoch)) '  ' num2str(alpha) ' test cost:' num2str(testCost) ] ); 
        end
       if isfield(options,'plot') && options.plot == 1, updatefigures(fhandle, [trainCst', testCst'], epoch, options.epochs); end  
       alpha = alpha * 0.99; 
    end
    if best, w = wOpt; end
end
