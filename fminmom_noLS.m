function [w, opt_epoch] = fminmom_noLS(f, fval, X, y, Xtest, ytest, w, options,  best, display, normalize)
    
    velocity = zeros(size(w));
    alpha = 0.01;
    momentum_i = 0.5;
    momentum_f = 0.9;
    num_batches = floor( size(X,1) / options.batchSize );
    minTestCost = Inf;
    batchSize = options.batchSize;
    wOpt = w;
    opt_epoch = 0;
    if isfield(options,'plot') && options.plot == 1, fhandle = figure(); end
    
    for epoch = 1:options.epochs,
        
        pos = randperm(size(X,1));
        X = X(pos,:);
        y = y(pos,:);
        
        if epoch < options.epochs / 2
                momentum = momentum_i + (momentum_f - momentum_i)*epoch*2/(options.epochs);
        else momentum = momentum_f;
        end
        
        for batch = 1:num_batches,
            
            curX = X( (batch-1)*batchSize+1 : batch*batchSize, : );
            curY = y( (batch-1)*batchSize+1 : batch*batchSize, : );
            
            [trainCost, grad] = f( w, curX, curY );
%             velocity = velocity * momentum - (1 - momentum)*alpha*grad;
            velocity = velocity * momentum - alpha*grad;
            w = w + velocity;
            
%             curNorm = norm(w);
%             if curNorm > maxNorm, w = w * 0.9 * maxNorm ./ curNorm; end%disp('********** divided by norm *********'); end 
%             w = normalize(w);
        end
        
        trainCst(epoch) = f( w, X, y );  
        testCst(epoch) = fval( w, Xtest, ytest ); 
        testCost = testCst(epoch);
        if isfield(options,'plot') && options.plot == 1, updatefigures(fhandle,[trainCst' testCst'],epoch,options.epochs); end
        if display, 
            disp( ['epoch ' num2str(epoch) ' trainCost:' num2str(trainCost) '  ' num2str(trainCostAf) '  ' num2str(alpha) ' test cost:' num2str(testCost) ] ); 
        end
        if best && testCost < minTestCost, minTestCost = testCost;   opt_epoch = epoch;    wOpt = w; end
         alpha = alpha * 0.995;
    end
    if best, w = wOpt; end
end
