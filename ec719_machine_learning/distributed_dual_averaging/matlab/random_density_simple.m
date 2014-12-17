function random_density_simple
% compares different properties of random graphs
close all;
n = 2500;
x = rand(2,n);
wtrue = [2 6]';
btrue = -4.5;
eta = 1./(1+exp(wtrue'*x + btrue*ones(1,n)));
y = binornd(ones(1,n),eta);

alphafun = @(t)(1 / t);
max_iter = 1000;

densrange = 0.25:0.1:0.95;
rsize = size(densrange, 2);
runs = 10;

iterhist = zeros(rsize, runs);
ll_hist = iterhist; % ugh, both ll_hist and llhist
nodes = 25;
for dens = 1:rsize
    for runi = 1:runs
        % generate state for distributed method
        [bigP, ax, ay] = gen_random(nodes, densrange(dens));
        
        if runi == 1
            % plot a sample random graph
            subplot(3, 3, dens);
            plot_random(ax, ay, bigP);
        end
        
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
        iterhist(dens, runi) = nodedata(1).iter;
        ll_hist(dens, runi) = llhist(end);   
    end
end

figure;
err = std(iterhist') / sqrt(nodes);
errorbar(densrange, mean(iterhist'), err);
xlabel('Communication distance');
ylabel('Iterations to convergence');
title('Effect of link density on random graph convergence speed');

end