function [A, x, y] = gen_random(n, r, side)    
    % generate a doubly stochastic matrix corresponding to 
    % a random connected geometric graph with number of nodes n 
    % and link radius r
    % (nodes are randomly distributed on a 2d [0,side] plane).
    
    % For a small radius / number of points, this can take a very long
    % time until a connected graph is generated.
    
    if ~exist('side', 'var'), side = 1; end
    
    [x, y, A, c] = gen_maybe_connected(n, r, side);
    while (~c)
        [x, y, A, c] = gen_maybe_connected(n, r, side);
    end
    
    % iteratively normalize the connectivity matrix to doubly stochastic
    diff = 1;
    while (diff > 0.0001)
        diff = 0;
        for iter = 1:5
            conv = zeros(1, 2*n);
            for i=1:n
                s = sum(A(:, i));
                diff = diff + (s - 1)^2;
                A(:, i) = A(:, i) ./ s;
            end
            for i=1:n
                s = sum(A(i, :));
                diff = diff + (s - 1)^2;
                A(i, :) = A(i, :) ./ s;
            end
        end
    end
end

function [x, y, A, c] = gen_maybe_connected(n, r, side)
    % throw some nodes on a plane
    pos = random('Uniform', 0, side, 2, n);
    x = pos(1, :); y = pos(2, :);

    % make a connectivity matrix
    A = zeros(n);
    for i = 1:n
        A(i, i) = 1;
        num_conn = 0;
        for j=i:n
            dx = abs(x(i) - x(j)); dy = abs(y(i) - y(j));
            if ((dx^2 + dy^2) < r^2)
                A(i, j) = 1;
                A(j, i) = 1;
                num_conn = num_conn + 1;
            end            
        end
    end

    % check connected status
    [p, q, r, s] = dmperm(A);
    c = (size(r, 2) == 2);
end