% Implement local (non-distributed) dual subgradient method on
% an example of logistic regression and compare to another optimization
% method (in this case, Newton-Raphson)

function local_simple
close all;
n = 5000;
x = rand(2,n);
wtrue = [2 6]';
btrue = -4.5;
eta = 1./(1+exp(wtrue'*x + btrue*ones(1,n)));
y = binornd(ones(1,n),eta);

xsc = x(:, 1:200);
ysc = y(1:200);

scatter(xsc(1,find(ysc==1)), xsc(2,find(ysc==1)), 'bo');
hold on;
scatter(xsc(1,find(ysc==0)), xsc(2,find(ysc==0)), 'rx');
hp0 = -btrue/wtrue(2);
hp1 = -(btrue+wtrue(1))/wtrue(2);
plot([0 1], [hp0 hp1], 'Linewidth', 2);
axis([0 1 0 1]);

newtraph(x, y, 0.001, 10000); % "NEWTon - RAPHson"
dualdescent(x, y, 0.001, 10000);
end

% a

function newtraph(x, y, precision, max_iter)
    % initializer
    nsamp = size(x, 2);
    x = [ones(1, nsamp); x]; % extend x with the intercept term
    theta = zeros(1, 3); % theta0/offset; theta1/x1; theta2/x2
    p = zeros(1, nsamp); % p(y_i) = 0
    ll_prev = -1000000;
    ll_delta = 1;
    iter = 1;
       
    while ll_delta > precision && iter < max_iter
        ll = 0;
        for i = 1:nsamp
            p(i) = 1 / (1 + exp( -theta * x(:, i)));
            ll = ll + y(i)*log(p(i)) + (1-y(i))*log(1-p(i));
        end
    
        hess = x * diag(p .* (p - ones(1, nsamp))) * x';
        grad = x * (y - p)';
        theta = theta - (0.2 * inv(hess) * grad)';
        
        ll_delta = ll - ll_prev;
        ll_prev = ll;
        iter = iter + 1;
    end
    fprintf('NR iterations = %d, ll=%f\n', iter, ll);
    theta

    hp0 = -theta(1)/theta(3);
    hp1 = -(theta(1)+theta(2))/theta(3);
    plot([0 1], [hp0 hp1], 'Linewidth', 2, 'Color', 'g');

    %fprintf('NR result: b = %f, w1 = %f, w2 = %f\n', theta(1), theta(2), theta(3));   
end

function dualdescent(x, y, precision, max_iter)
    [d, n] = size(x);    
    x = [ones(1, n); x]; % extend x with the intercept term
    d = d + 1;
    theta = zeros(d, 1); % initial values for theta0/offset; theta1/x1; theta2/x2
    alphafun = @(t)(1 / t);
    llprev = 1e9;
    lldelta = 1;
    theta_hist = theta;
    
    iter = 1;
    daz = zeros(d, 1); % dual averaging initial iterates    
    while lldelta > precision && iter < max_iter
        p = 1 ./ (1 + exp(theta' * x));
        p(p < 1e-7) = 1e-7;
        p(p > 1-1e-7) = 1-1e-7;
        ll = sum(y .* log(p) + (1-y) .* log(1-p));
        
        grad = x * (p - y)';
        daz = daz - grad;
        alpha = alphafun(iter);
        theta = (-alpha / 2) * daz;
        
        lldelta = abs(ll - llprev);
        llprev = ll;
        theta_hist = [theta_hist theta];
        iter = iter + 1;  
    end
    fprintf('DD iterations = %d, ll = %f\n', iter, ll);
    theta'
    
    hp0 = -theta(1)/theta(3);
    hp1 = -(theta(1)+theta(2))/theta(3);
    plot([0 1], [hp0 hp1], 'Linewidth', 2, 'Color', 'r');
    
    title('Dual averaging descent compared to Newton-Raphson');
    legend('', '', 'Actual parameters', 'N-R estimation', 'Dual averaging estimation');
    
    figure;
    plot(theta_hist');
    axis([0 80 -20 20]);
    title('Convergence of dual averaging descent');
    legend('theta(0)', 'theta(1)', 'theta(2)');
    xlabel('Iterations');
    ylabel('Values');
end