function [J, grad] = costFunction_3more_2hid( nn_params, ...
                                   input_layer1_size, ...
                                   input_layer2_size, ...
                                   input_layer3_size, ...
                                   hidden_layer11_size, ...
                                   hidden_layer12_size, ...
                                   hidden_layer13_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X1, X2, X3, y, lambda, mean1, display )


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
    a21 = [ ones(m,1) tanh_opt( a11 * Theta11' ) ];
    a22 = tanh_opt( a12 * Theta12' );
    a23 = tanh_opt( a13 * Theta13' );
    a2 = [a21 a22 a23];
    a3 = [ ones(m,1) tanh_opt( (a2 - repmat( [0 mean1], m, 1 )) * Theta2' ) ];
    a4 = a3 * Theta3';

    J = mean(sum( (a4 - y).^2, 2 ))/2;
    
    if exist('display','var')
        disp('entered cost function')
        disp('predicted')
        disp(a4)
        disp('desired')
        disp(y)
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    del_4 = ( a4 - y );
    del_3 = ( del_4 * Theta3 ) .* ( 1.7159 * (2/3) * ( 1 - (a3.^2)/(1.7159^2) ) );
    del_3 = del_3(:,2:end);
    del_2 = ( del_3 * Theta2 ) .* ( 1.7159 * (2/3) * ( 1 - (a2.^2)/(1.7159^2) ) );
    del_2 = del_2(:,2:end);
    del_21 = del_2(:,1 : hidden_layer11_size);
    del_22 = del_2(:,hidden_layer11_size+1 : hidden_layer12_size+hidden_layer11_size);
    del_23 = del_2(:,hidden_layer11_size+hidden_layer12_size+1 : end);
%     gradTheta_1 = (lambda * [zeros(size(Theta1,1),1) Theta1(:,2:end)]) + ( del_2' * a1 )/m;
%     gradTheta_2 = (lambda * [zeros(size(Theta2,1),1) Theta2(:,2:end)]) + ( del_3' * a2 )/m;
%     gradTheta_3 = (lambda * [zeros(size(Theta3,1),1) Theta3(:,2:end)]) + ( del_4' * a3 )/m;
    
    
    c = 0.81;
    gradTheta_11 = (lambda * [ zeros(size(Theta11,1),1) (2*c*Theta11(:,2:end)) ./ ((c+Theta11(:,2:end).^2).^2) ]) + ( del_21' * a11 )/m;
    gradTheta_12 = (lambda * [ zeros(size(Theta12,1),1) (2*c*Theta12(:,2:end)) ./ ((c+Theta12(:,2:end).^2).^2) ]) + ( del_22' * a12 )/m;
    gradTheta_13 = (lambda * [ zeros(size(Theta13,1),1) (2*c*Theta13(:,2:end)) ./ ((c+Theta13(:,2:end).^2).^2) ]) + ( del_23' * a13 )/m;
    gradTheta_2 = (lambda * [ zeros(size(Theta2,1),1) (2*c*Theta2(:,2:end)) ./ ((c+Theta2(:,2:end).^2).^2) ]) + ( del_3' * (a2 - repmat( [0 mean1], size(a2,1), 1 )) )/m;
    gradTheta_3 = (lambda * [ zeros(size(Theta3,1),1) (2*c*Theta3(:,2:end)) ./ ((c+Theta3(:,2:end).^2).^2) ]) + ( del_4' * a3 )/m;
    
    
%     gradTheta_1 = (lambda * [ zeros(size(Theta1,1),1) sign(Theta1(:,2:end))  ]) + ( del_2' * a1 )/m;
%     gradTheta_2 = (lambda * [ zeros(size(Theta2,1),1) sign(Theta2(:,2:end))  ]) + ( del_3' * (a2 - repmat( [0 mean1], size(a2,1), 1 )) )/m;
%     gradTheta_3 = (lambda * [ zeros(size(Theta3,1),1) sign(Theta3(:,2:end))  ]) + ( del_4' * a3 )/m;
    
    grad = [gradTheta_11(:); gradTheta_12(:); gradTheta_13(:); gradTheta_2(:); gradTheta_3(:)];

end
