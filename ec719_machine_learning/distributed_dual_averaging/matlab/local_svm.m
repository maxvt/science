function local_svm
close all;
load mnist_49_8x8;

[d,n] = size(x);

xtrain = x(:, 1:2500);
ytrain = y(1:2500);

xtest = x(:, 2501:end);
ytest = y(2501:end);

c = dualdescent(xtrain, ytrain, 0.0001, 500, 1000, @(a,b)(a' * b + 1));
samp = num2cell(xtest, 1);
est = cellfun(c, samp); 
err = sum(est ~= ytest);
fprintf('test error = %f\n', err / size(ytest, 2));   
end

function classifier = dualdescent(x, y, precision, max_iter, bigC, k)
    [d, n] = size(x);
    theta = zeros(n, 1); % call dual svm alphas "thetas" to avoid confusion
                         % with dual descent "alpha" sequence.
    alphafun = @(t)(0.1 / sqrt(t));
    objprev = 0;
    objdelta = 1;
    theta_hist = theta;
    
    iter = 1;
    daz = zeros(n, 1); % dual averaging initial iterates
    K = k(x, x);
    
    while objdelta > precision && iter < max_iter       
        grad = zeros(n, 1);
        for i=1:n
            t = 1/2 * y(i) * (sum(theta' .* y .* K(i, :)) - theta(i) * y(i) * K(i, i));
            grad(i) = (1 - t) / K(i, i);
        end
        
        daz = daz - grad;
        alpha = alphafun(iter);
        theta = (-alpha / 2) * daz;
        theta(theta < 0) = 0;
        theta(theta > (bigC / n)) = bigC / n;
        
        objval = - sum(theta) + 1/2 * (theta' .* y) * K * (theta .* y')       
        objdelta = abs(objval - objprev);
        objprev = objval;
        theta_hist = [theta_hist theta];
        iter = iter + 1;  
    end
    fprintf('DD iterations = %d\n', iter);
    theta = theta';
    sv = theta > 0;
    fprintf('num_sv = %d\n', sum(sv));
    classifier = @(samp)(-1 + 2 * ((theta(sv) .* y(sv) * k(x(:, sv), samp)) > 0));
end