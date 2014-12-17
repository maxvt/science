function tune_lpn
noderange = 10:10:90;
rsize = size(noderange, 2);
runs = 50;
for nodei = 1:rsize
    tlinks = 0;
    for runi = 1:runs
        nodes = noderange(nodei);
        [bigP, ax, ay] = gen_random(nodes, 0.8, (nodes/10)^(0.67));
        tlinks = tlinks + sum(sum(bigP>0))/2;        
    end
    lpn = tlinks / (nodes * runs);
    fprintf('n=%d, lpn=%f\n', nodes, lpn);
end
end