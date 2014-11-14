function J = valcostFunction_do( nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, dropOutRatios )

    if ~exist('dropOutRatios','var')
        dropOutRatios = [0 0];
    end
%     assert( size(dropOutRatios,1) == 1 && size(dropOutRatios,2) == 1, 'dropOutRatios size is improper');
                               
    m = size(X,1);
    Theta1 = reshape( nn_params(1:hidden_layer_size*(input_layer_size+1)) , hidden_layer_size , input_layer_size + 1);
    Theta2 = reshape( nn_params(hidden_layer_size*(input_layer_size+1)+1:end), num_labels, hidden_layer_size+1 );

    X = X .* ( 1 - dropOutRatios(1) );
    a1 = [ ones(m,1) X ];
    a2 = tanh_opt( a1 * Theta1' );
    a2 = a2 .* ( 1 - dropOutRatios(2) );
    a2 = [ ones(m,1) a2 ];
    a3 = a2 * Theta2';

    J = mean(sum( (a3 - y).^2, 2 ))/2;

end