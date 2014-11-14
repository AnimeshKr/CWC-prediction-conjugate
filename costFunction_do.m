function [J, grad] = costFunction( nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda, dropOutRatios )

    if ~exist('dropOutRatios','var')
        dropOutRatios = [0 0];
    end
%     assert( size(dropOutRatios,1) == 1 && size(dropOutRatios,2) == 1, 'dropOutRatios size is improper');
                               
    m = size(X,1);
    Theta1 = reshape( nn_params(1:hidden_layer_size*(input_layer_size+1)) , hidden_layer_size , input_layer_size + 1);
    Theta2 = reshape( nn_params(hidden_layer_size*(input_layer_size+1)+1:end), num_labels, hidden_layer_size+1 );
    
    dropOutMask{1} = rand(size(X)) >= dropOutRatios(1);
    X = X .* dropOutMask{1};
    a1 = [ ones(m,1) X ];
    a2 = tanh_opt( a1 * Theta1' );
    dropOutMask{2} = rand(size(a2)) >= dropOutRatios(2);
    a2 = a2 .* dropOutMask{2};
    a2 = [ ones(m,1) a2 ];
    a3 = a2 * Theta2';

    J = mean(sum( (a3 - y).^2, 2 ))/2;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%gradient%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    del_3 = ( a3 - y );
    del_2 = ( del_3 * Theta2 ) .* ( 1.7159 * (2/3) * ( 1 - (a2.^2)/(1.7159^2) ) ); 
    del_2 = del_2(:,2:end);
    del_2 = del_2 .* dropOutMask{2};

%     gradTheta_1 = (lambda * [zeros(size(Theta1,1),1) Theta1(:,2:end)]) + ( del_2' * a1 )/m;
%     gradTheta_2 = (lambda * [zeros(size(Theta2,1),1) Theta2(:,2:end)]) + ( del_3' * a2 )/m;


    c = 0.81;
    gradTheta_1 = ( lambda * [ zeros(size(Theta1,1),1) (2*c*Theta1(:,2:end)) ./ ((c+Theta1(:,2:end).^2).^2) ] ) + ( del_2' * a1 )/m;
    gradTheta_2 = ( lambda * [ zeros(size(Theta2,1),1) (2*c*Theta2(:,2:end)) ./ ((c+Theta2(:,2:end).^2).^2) ] ) + ( del_3' * a2 )/m;


%     gradTheta_1 = (lambda * [zeros(size(Theta1,1),1) sign(Theta1(:,2:end)) ]) + ( del_2' * a1 )/m;
%     gradTheta_2 = (lambda * [zeros(size(Theta2,1),1) sign(Theta2(:,2:end)) ]) + ( del_3' * a2 )/m;

    grad = [gradTheta_1(:); gradTheta_2(:)];

end% derivative of tanH fuction (If we change the activation function to some other function than tanH, we need to replace this by the corresponding function derivative)