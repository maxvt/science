function dist_simple
% compares a distributed cycle and grid of the same node count.
close all;
n = 16000;
x = rand(2,n);
wtrue = [2 6]';
btrue = -4.5;
eta = 1./(1+exp(wtrue'*x + btrue*ones(1,n)));
y = binornd(ones(1,n),eta);

max_iter = 1000;

% generate state for distributed method
nodes = 16;
bigP = gen_cycle(nodes);
oldzvec = zeros(3, nodes);
samp_per_node = n/nodes;
sumconverged = 0;
% alphafun = @(t)(1 / t);
% R = 6, L = 1 (data are in 0..1)
alphafun = @(t)(1 / t);
for i = 1:nodes
    nodex = x(:, 1+samp_per_node*(i-1) : samp_per_node*i);
    nodey = y(1+samp_per_node*(i-1) : samp_per_node*i);
    cycledata(i) = distddinit(nodex, nodey, 0.001, 10000, alphafun, bigP(i, :));
end
cllhist = zeros(nodes, 1);

% run distributed method
while (cycledata(1).iter < max_iter && sumconverged < nodes)
    newzvec = zeros(3, nodes);
    sumconverged = 0;
    ll = zeros(i, 1);
    for i = 1:nodes
        [cycledata(i), newz, converged] = distddstep(cycledata(i), oldzvec);
        sumconverged = sumconverged + converged;
        newzvec(:, i) = newz;
        ll(i) = cycledata(i).llprev;
    end
    cllhist = [cllhist ll];
    oldzvec = newzvec;
end

fprintf('cycle graph iterations = %d, avg ll = %f\n', cycledata(1).iter, mean(ll));

% generate state for distributed method
nodes = 16;
bigP = gen_grid(nodes);
oldzvec = zeros(3, nodes);
samp_per_node = n/nodes;
sumconverged = 0;
alphafun = @(t)(0.04 / sqrt(t));
for i = 1:nodes
    nodex = x(:, 1+samp_per_node*(i-1) : samp_per_node*i);
    nodey = y(1+samp_per_node*(i-1) : samp_per_node*i);
    griddata(i) = distddinit(nodex, nodey, 0.001, 10000, alphafun, bigP(i, :));
end
gllhist = zeros(nodes, 1);

% run distributed method
while (griddata(1).iter < max_iter && sumconverged < nodes)
    newzvec = zeros(3, nodes);
    sumconverged = 0;
    ll = zeros(i, 1);
    for i = 1:nodes
        [griddata(i), newz, converged] = distddstep(griddata(i), oldzvec);
        sumconverged = sumconverged + converged;
        newzvec(:, i) = newz;
        ll(i) = griddata(i).llprev;
    end
    gllhist = [gllhist ll];
    oldzvec = newzvec;
end

fprintf('grid graph iterations = %d, avg ll = %f\n', griddata(1).iter, mean(ll));

for i = 1:16
    subplot(4, 4, i);
    semilogy(cllhist(i, :), 'b');
    hold on;
    semilogy(gllhist(i, :), 'r');
    axis([1 400 -5000 -100]);
    %title('Comparison of different graph types');
    %legend('Cycle', '2D Grid');
    %xlabel('Iterations');
    %ylabel('Mean log-likelihood');
end

figure;
semilogy(mean(cllhist), 'b'); hold on; semilogy(mean(gllhist), 'r');
title('Mean log-likelihood over all nodes');
legend('Cycle', '2D Grid');
xlabel('Iterations');
ylabel('Mean log-likelihood');
axis([0 200 -10000 -200]);
end