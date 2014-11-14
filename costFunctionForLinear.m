function [J, grad] = costFunctionForLinear( nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda )
                               
                               
    [m, n] = size(X);
    Theta1 = reshape( nn_params(1:hidden_layer_size*(input_layer_size+1)) , hidden_layer_size , input_layer_size + 1);
    Theta2 = reshape( nn_params(hidden_layer_size*(input_layer_size+1)+1:end), num_labels, hidden_layer_size+1 );
    
    a1 = [ ones(m,1) X ];
    a2 = [ ones(m,1) ( a1 * Theta1' ) ];
    a3 = a2 * Theta2';

    J = mean(sum( (a3 - y).^2, 2 ))/2;

    
    del_3 = ( a3 - y );
    del_2 = ( del_3 * Theta2 ) ;
    del_2 = del_2(:,2:end);

    gradTheta_1 = ((lambda) * [zeros(size(Theta1,1),1) Theta1(:,2:end)]) + ( del_2' * a1 )/m;
    gradTheta_2 = ((lambda) * [zeros(size(Theta2,1),1) Theta2(:,2:end)]) + ( del_3' * a2 )/m;
    grad = [gradTheta_1(:); gradTheta_2(:)];

end