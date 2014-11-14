function [J, grad] = costFunction_2hid( nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, y, lambda )


    [m, ~] = size(X);
    Theta1 = reshape( nn_params(1:hidden_layer1_size*(input_layer_size+1)), hidden_layer1_size, input_layer_size + 1);
    Theta2 = reshape( nn_params(hidden_layer1_size*(input_layer_size+1)+1: hidden_layer1_size*(input_layer_size+1) + hidden_layer2_size*(hidden_layer1_size+1)), hidden_layer2_size, hidden_layer1_size+1 );
    Theta3 = reshape( nn_params( end - (num_labels*(hidden_layer2_size+1)-1) :end), num_labels, hidden_layer2_size+1 );

    
    a1 = [ ones(m,1) X ];
    a2 = [ ones(m,1) tanh_opt( a1 * Theta1' ) ];
    a3 = [ ones(m,1) tanh_opt( a2 * Theta2' ) ];
    a4 = a3 * Theta3';

    J = mean(sum( (a4 - y).^2, 2 ))/2;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    del_4 = ( a4 - y );
    del_3 = ( del_4 * Theta3 ) .* ( 1.7159 * (2/3) * ( 1 - (a3.^2)/(1.7159^2) ) );
    del_3 = del_3(:,2:end);
    del_2 = ( del_3 * Theta2 ) .* ( 1.7159 * (2/3) * ( 1 - (a2.^2)/(1.7159^2) ) );
    del_2 = del_2(:,2:end);
    
    gradTheta_1 = ((lambda) * [zeros(size(Theta1,1),1) Theta1(:,2:end)]) + ( del_2' * a1 )/m;
    gradTheta_2 = ((lambda) * [zeros(size(Theta2,1),1) Theta2(:,2:end)]) + ( del_3' * a2 )/m;
    gradTheta_3 = ((lambda) * [zeros(size(Theta3,1),1) Theta3(:,2:end)]) + ( del_4' * a3 )/m;
    grad = [gradTheta_1(:); gradTheta_2(:); gradTheta_3(:)];

end
