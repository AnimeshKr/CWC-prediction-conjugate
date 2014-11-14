%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function computes cost function for the final archietecture, performs back propagation and returns gradient





function [J, grad] = valcostFunction_3more_2hid_do( nn_params, ...
                                   input_layer1_size, ...
                                   input_layer2_size, ...
                                   input_layer3_size, ...
                                   hidden_layer11_size, ...
                                   hidden_layer12_size, ...
                                   hidden_layer13_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X1, X2, X3, y, mean1, dropOutRatios)


    if ~exist('dropOutRatios','var'),
        dropOutRatios = [0 0];
    end
    
    
    [m, ~] = size(X1);
    assert(size(X1,1) == size(X2,1) , 'dimensions not matching');

    no_11 = hidden_layer11_size * (input_layer1_size+1);
    no_12 = hidden_layer12_size * (input_layer2_size+1);
    no_13 = hidden_layer13_size * (input_layer3_size+1);
    no_2 = hidden_layer2_size * (hidden_layer11_size + hidden_layer12_size + hidden_layer13_size+1);
    
    
    Theta11 = reshape( nn_params(1 : no_11), hidden_layer11_size, input_layer1_size+1 );
    Theta12 = reshape( nn_params(no_11+1 : no_11+no_12), hidden_layer12_size, input_layer2_size+1 );
    Theta13 = reshape( nn_params(no_11+no_12+1 : no_11+no_12+no_13), hidden_layer13_size, input_layer3_size+1 );
    Theta2  = reshape( nn_params(no_11+no_12+no_13+1 : no_11+no_12+no_13+no_2), hidden_layer2_size, hidden_layer11_size+hidden_layer12_size+hidden_layer13_size+1 );
    Theta3  = reshape( nn_params(no_11+no_12+no_13+no_2+1 : end), num_labels, hidden_layer2_size+1 );
    
    
    a11 = [ ones(m,1) X1 ];
    a12 = [ ones(m,1) X2 ];
    a13 = [ ones(m,1) X3 ];
    a21 = tanh_opt( a11 * Theta11' );
    a22 = tanh_opt( a12 * Theta12' );
    a23 = tanh_opt( a13 * Theta13' );
    a2 = [a21 a22 a23];
    a2 = a2 .* ( 1 - dropOutRatios(1) );
    a2 = [ ones(m,1) a2 ]; 
    a3 =  tanh_opt( (a2 - repmat( [0 mean1], m, 1 )) * Theta2' );
    a3 = a3 .* ( 1 - dropOutRatios(2) );
    a3 = [ ones(m,1) a3 ];
    a4 = a3 * Theta3';
    
    
     %% mse error (1/2*(summation of square of error in prediction)
    J = mean(sum( (a4 - y).^2, 2 ))/2;
     
%     if exist('display','var')
%         disp('entered cost function')
%         disp('predicted')
%         disp(a4)
%         disp('desired')
%         disp(y)
%     end

    
end
