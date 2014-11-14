clear all;
clc;

%%Loading input data
load input_cov.mat;
randpos = randperm(size(X,2));
X = X(:,randpos);
save('featuresPerm.mat','randpos');

%%Loading output data
load output_south.mat;
y = 100*y;

origX = X;
origY = y;
meanValues = mean(origY);
assert(size(X,1) == size(y,1) , 'Dimensions of input and output are not matching');

firstIn = floor(size(X,2)/3);
secondIn = floor(2*size(X,2)/3);
    
%%%%%%%Dividing before seperation
startYear = 2007;
endYear = 2008;
prevYears = 60;

for i = 1:(endYear-startYear)+1,

    predictionYear = startYear + i - 1;
    errors(i,1) = predictionYear;
    prevErrors(i,1) = predictionYear;

    Xtest = origX(end - ( (2011-predictionYear)*122+121 ) : end - (2011-predictionYear)*122 , :);
    ytest = origY(end - ( (2011-predictionYear)*122+121 ) : end - (2011-predictionYear)*122 , :);
    
%     curYearX = origX( end - ( ( (2011+prevYears)-predictionYear )*122+121 ) : end - ( (2011-predictionYear)*122+122 ), : );
%     curYearY = origY( end - ( ( (2011+prevYears)-predictionYear )*122+121 ) : end - ( (2011-predictionYear)*122+122 ), : );
     
    curYearX = origX( 1 : end - ( (2011-predictionYear)*122+122 ), : );
    curYearY = origY( 1 : end - ( (2011-predictionYear)*122+122 ), : );
    
    
    initAvg = mean(curYearX);
    curYearX = curYearX - repmat(initAvg, size(curYearX,1), 1);
    Xtest = Xtest - repmat(initAvg, size(Xtest,1), 1);
    initRan = std(curYearX);
    curYearX = curYearX ./ repmat(initRan, size(curYearX,1), 1);
    Xtest = Xtest ./ repmat(initRan, size(Xtest,1), 1);
    
    X1test = Xtest(:,1:firstIn);
    X2test = Xtest(:,firstIn+1:secondIn);
    X3test = Xtest(:,secondIn+1:end);

    curYearX1 = curYearX(:,1:firstIn);
    curYearX2 = curYearX(:,firstIn+1:secondIn);
    curYearX3 = curYearX(:,secondIn+1:end);

    sae1 = train_1part_AE(curYearX1,X1test,0.02,0.2,0.25);
    sae2 = train_1part_AE(curYearX2,X2test,0.02,0.2,0.25);
    sae3 = train_1part_AE(curYearX3,X3test,0.02,0.2,0.25);
    
    m = size(curYearX,1);
    curYearfeatures = [ tanh_opt( [ ones(m,1) curYearX1 ] * sae1.Theta1' )  tanh_opt( [ ones(m,1) curYearX2 ] * sae2.Theta1' )  tanh_opt( [ ones(m,1) curYearX3 ] * sae3.Theta1' )];
    mtest = size(Xtest,1);
    featuresTest = [ tanh_opt( [ ones(mtest,1) X1test ] * sae1.Theta1' )  tanh_opt( [ ones(mtest,1) X2test ] * sae2.Theta1' )  tanh_opt( [ ones(mtest,1) X3test ] * sae3.Theta1' )];
    mean(curYearfeatures)
%     mean(featuresTest)
%     pause;
    
    input_layer_size = size(curYearfeatures,2);
    hidden_layer_size = 40;
    num_labels = size(curYearY,2);
    lambda = 0.05;
    
    for exp = 1:1,
        disp( ['Predicting year ' num2str(predictionYear) ' experiment number ' num2str(exp)] );

        costFunc = @(p) costFunction(p, ...
                                           input_layer_size, ...
                                           hidden_layer_size, ...
                                           num_labels, curYearfeatures, curYearY, lambda);

        valcostFunc = @(p) costFunction(p, ...
                                           input_layer_size, ...
                                           hidden_layer_size, ...
                                           num_labels, featuresTest, ytest, 0);

        initial_Theta1 = [zeros(hidden_layer_size,1) (rand( hidden_layer_size, input_layer_size ) - 0.5 ) * 4 * sqrt(6 / (hidden_layer_size+input_layer_size))] ;
        initial_Theta2 = [zeros(num_labels,1) (rand( num_labels, hidden_layer_size ) - 0.5) * 10 * sqrt(6 / (hidden_layer_size+num_labels))] ;
        nn_params =[ initial_Theta1(:); initial_Theta2(:) ] ;

        options = optimset('MaxIter', 200);
        [nn_params, ~] = fmincg(costFunc, nn_params, options, valcostFunc, 1, 1);

        % Obtain Theta1 and Theta2 back from nn_params
        nn.Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                         hidden_layer_size, (input_layer_size + 1));

        nn.Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                         num_labels, (hidden_layer_size + 1));
    
        a2 = tanh_opt([ones(size(featuresTest,1),1) featuresTest] * nn.Theta1');
        a3 = [ones(size(featuresTest,1),1) a2] * nn.Theta2';
        a3
        ytest
        mean( sum((a3-ytest).^2, 2 ) )/2
        mean(a2)
        pause;
        [~, prevErrors(i,2*exp), ~, prevErrors(i,2*exp+1)] = feedForward_superVised(nn,curYearfeatures,curYearY,featuresTest,ytest,meanValues);

        disp('test rms error');
        disp(prevErrors(i,2*exp));
        disp('test abs error');
        disp(prevErrors(i,2*exp+1));
        mean1 = zeros(1,size(curYearfeatures,2));
        [finalNN, ~, errors(i, 3*exp-1), ~,errors(i, 3*exp), errors(i, 3*exp+1)] = fineTune_3more_noVal_lib(nn, sae1, sae2, sae3, curYearX1, curYearX2, curYearX3, curYearY, X1test, X2test, X3test, ytest, meanValues, mean1, predictionYear);
    end  
end

disp('previous rms error');
disp(mean(mean( prevErrors(:,2:2:end) )))

disp('previous abs error');
disp(mean(mean( prevErrors(:,3:2:end) )))

disp('fine tuned rms error');
disp(mean(mean(errors(:,2:3:3*exp+1))))

disp('fine tuned abs error');
disp(mean(mean(errors(:,3:3:3*exp+1))))

disp('fine tuned mean min error');
disp(mean(mean(errors(:,4:3:3*exp+1))))
    

  csvwrite( ['3autowithoutvalprevErrors' num2str(prevYears) 'from' num2str(startYear) 'to' num2str(endYear) 'hid' num2str(hidden_layer_size) 'lambda' num2str(lambda) '.csv'], prevErrors );
  csvwrite( ['3autowithoutvalErrors' num2str(prevYears) 'from' num2str(startYear) 'to' num2str(endYear) 'hid' num2str(hidden_layer_size) 'lambda' num2str(lambda) '.csv'], errors );