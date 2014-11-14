function [w] = fminmom_reg(f, fval, X, y, Xtest, ytest, w, options,  best, display, normalize)
    
    velocity = zeros(size(w));
    alpha_start = 0.3;
    momentum_i = 0.4;
    momentum_f = 0.9;
    num_batches = floor( size(X,1) / options.batchSize );
    minTestCost = Inf;
    batchSize = options.batchSize;
    wOpt = w;
    
    iter = 1;
    for epoch = 1:options.epochs,
        
        pos = randperm(size(X,1));
        X = X(pos,:);
        y = y(pos,:);
        
        if iter < options.epochs / 2
                momentum = momentum_i + (momentum_f - momentum_i)*iter*2/(options.epochs);
        end
        
        for batch = 1:num_batches,
            
            curX = X( (batch-1)*batchSize+1 : batch*batchSize, : );
            curY = y( (batch-1)*batchSize+1 : batch*batchSize, : );
            
            
            [trainCost, grad] = f( w, curX, curY );
%             velocity = velocity * momentum - (1 - momentum)*alpha*grad;
%             w = w + velocity;
            
            alpha = alpha_start;
            i = 1;
            while i < 30
                trainCostAf = f( w + velocity*momentum - alpha*grad, curX, curY );
                if trainCostAf < trainCost - 0.01*alpha*( grad' * (velocity*momentum - alpha*grad) ),
                    break;
                end
                alpha = alpha / 1.5;
                i = i + 1;
            end
            if i < 30,
                velocity = velocity * momentum - alpha*grad;
                w = w + velocity;
            end
            trainCostAf = f( w, curX, curY );
            
%             curNorm = norm(w);
%             if curNorm > maxNorm, w = w * 0.9 * maxNorm ./ curNorm; end%disp('********** divided by norm *********'); end 
            w = normalize(w);
            testCost = fval( w, Xtest, ytest ); 
            if display, disp( ['epoch ' num2str(epoch) ' trainCost:' num2str(trainCost) '  ' num2str(trainCostAf) '  ' num2str(alpha) ' test cost:' num2str(testCost) ] ); end
            if best && testCost < minTestCost, minTestCost = testCost;  wOpt = w; end
            iter = iter + 1; 
        end
        
%         if epoch < 200
%             alpha = alpha * 1.004;
%         else
%             alpha = alpha * 0.998;
%         end
        
    end
    if best, w = wOpt; end
    
end