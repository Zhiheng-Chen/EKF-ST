function x = sparseRegression(A,b,params_regression)
%Zhiheng Chen
%10/30/2024
%three algorithms for sparse regressions
switch params_regression.algorithm
    case 'least-squares'
        lambda_sparse = params_regression.lambda_sparse;   %sparsification knob
        x = pinv(A)*b;   %least-squares regression
        for ii = 1:params_regression.N_loops
            smallInds = abs(x)<lambda_sparse;
            x(smallInds) = 0;
            bigInds = ~smallInds;
            x(bigInds) = pinv(A(:,bigInds))*b;
        end
    case 'ridge'
        lambda_sparse = params_regression.lambda_sparse;    %sparsification knob
        lambda_ridge = params_regression.lambda_ridge;   %ridge regression regulation multiplier
        x = (A'*A+lambda_ridge*eye(size(A'*A)))\(A'*b);
        for ii = 1:params_regression.N_loops
            smallInds = abs(x)<lambda_sparse;
            x(smallInds) = 0;
            bigInds = ~smallInds;
            A_bigInds = A(:,bigInds);
            x(bigInds) = (A_bigInds'*A_bigInds+lambda_ridge*eye(size(A_bigInds'*A_bigInds)))\(A_bigInds'*b);
        end
    case 'lasso'
        lambda_lasso = params_regression.lambda_lasso; %lasso regression regulation multiplier
        x = lasso(A,b,'lambda',lambda_lasso);
end