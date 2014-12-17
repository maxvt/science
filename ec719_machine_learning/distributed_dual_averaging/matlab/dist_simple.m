function dist_simple
% compares a single (centralized) run of dual averaging vs. distributed
close all;
n = 9000;
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

alphafun = @(t)(1 / t);
max_iter = 10000;
[c_theta, c_theta_hist, c_iter, c_ll, c_llhist] = dualdescent(x(:,1:1000), y(1:1000), alphafun, 0.001, max_iter);

     hp0 = -c_theta(1)/c_theta(3);
     hp1 = -(c_theta(1)+c_theta(2))/c_theta(3);
     plot([0 1], [hp0 hp1], 'Linewidth', 2, 'Color', 'r');
   
% generate state for distributed method
nodes = 9;
bigP = gen_cycle(nodes);
oldzvec = zeros(3, nodes);
samp_per_node = n/nodes;
sumconverged = 0;
for i = 1:nodes
    nodex = x(:, 1+samp_per_node*(i-1) : samp_per_node*i);
    nodey = y(1+samp_per_node*(i-1) : samp_per_node*i);
    nodedata(i) = distddinit(nodex, nodey, 0.001, 10000, alphafun, bigP(i, :));
end
llhist = zeros(nodes, 1);

% run distributed method
while (nodedata(1).iter < max_iter && sumconverged < nodes)
    newzvec = zeros(3, nodes);
    sumconverged = 0;
    ll = zeros(i, 1);
    for i = 1:nodes
        [nodedata(i), newz, converged] = distddstep(nodedata(i), oldzvec);
        sumconverged = sumconverged + converged;
        newzvec(:, i) = newz;
        ll(i) = nodedata(i).llprev;
    end
    llhist = [llhist ll];
    oldzvec = newzvec;
end

theta = nodedata(1).theta;
hp0 = -theta(1)/theta(3);
hp1 = -(theta(1)+theta(2))/theta(3);
plot([0 1], [hp0 hp1], 'Linewidth', 2, 'Color', 'g');
legend('', '', 'Actual parameters', 'Dual averaging', 'Distributed DA');

% calculate ll over all data for comparison, using distributed theta
p = 1 ./ (1 + exp(nodedata(1).theta' * [ones(1, n); x]));
p(p < 1e-7) = 1e-7; % avoid NaN in ll calculation
p(p > 1-1e-7) = 1-1e-7;
ll = sum(y .* log(p) + (1-y) .* log(1-p));

fprintf('distributed DD iterations = %d, avg ll = %f\n', nodedata(1).iter,ll);

% plot of evolution of estimation of theta(1) across all nodes & non-distr.
data = ones(1, nodedata(1).iter) * c_theta_hist(1, c_iter);
data(1:c_iter) = c_theta_hist(1, :);

for i = 1:3
    data = [data;nodedata(i).theta_hist(1, :)];
end
figure;
plot(data');
axis([0 150 -30 20]);
title('Convergence of single-node vs. distributed DA descent');
legend('Single-node', 'distributed, node 1', 'distributed, node 2', 'distributed, node 3');
xlabel('Iterations');
ylabel('Values');

figure;
data = ones(1, nodedata(1).iter) * c_llhist(1, c_iter);
data(1:c_iter) = c_llhist(1, :);
for i = 1:nodes
    data = [data;nodedata(i).llhist(1, :)];
end
avgdata = [data(1, 1:200); mean(data(2:end, 1:200))];
semilogy(avgdata');
title('Log-likelihood behavior');
xlabel('Iterations');
ylabel('Log-likelihood');
legend('Single-node', 'Distributed');
end

function [theta, theta_hist, iter, ll, llhist] = dualdescent(x, y, alphafun, precision, max_iter)
    [d, n] = size(x);    
    x = [ones(1, n); x]; % extend x with the intercept term
    d = d + 1;
    theta = zeros(d, 1); % initial values for theta0/offset; theta1/x1; theta2/x2    
    llprev = 1e9;
    lldelta = 1;
    llhist = 0;
    theta_hist = theta;
    
    iter = 1;
    daz = zeros(d, 1); % dual averaging initial iterates    
    while lldelta > precision && iter < max_iter
        p = 1 ./ (1 + exp(theta' * x));
        p(p < 1e-7) = 1e-7; % avoid NaN in ll calculation
        p(p > 1-1e-7) = 1-1e-7;
        ll = sum(y .* log(p) + (1-y) .* log(1-p));
        
        grad = x * (p - y)';
        daz = daz - grad;
        alpha = alphafun(iter);
        theta = (-alpha / 2) * daz;
        
        lldelta = abs(ll - llprev);
        llprev = ll;
        llhist = [llhist ll];
        theta_hist = [theta_hist theta];
        iter = iter + 1;  
    end
    fprintf('single-node DD iterations = %d, ll = %f\n', iter, ll);
end