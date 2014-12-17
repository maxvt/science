function A = gen_grid(n)
    % generate a doubly stochastic matrix for 2d wrapping grid of size n
    % (for non-wrapping grid, a doubly stochastic variant seems to be
    % complicated to make)
    
    %  | | |
    % -1-2-3-
    %  | | |
    % -4-5-6-
    %  | | |   
    % -7-8-9-
    %  | | |
       
    if (n == 4) 
        A = gen_cycle(4);
        return;
    end

    d = sqrt(n);
    c_in = 0.2;
    c_out = (1 - c_in) / 4;
    A = eye(n) * c_in;
    for i = 1:n
        if (i - d > 0)
            A(i, i-d) = c_out;
        else 
            A(i, d*(d-1)+i) = c_out;
        end
        if (i + d <= n)
            A(i, i+d) = c_out;
        else
            A(i, i-d*(d-1)) = c_out;
        end
        if (mod(i, d) ~= 1)
            A(i, i-1) = c_out;
        else
            A(i, i+d-1) = c_out;
        end
        if (mod(i, d) > 0)
            A(i, i+1) = c_out;
        else
            A(i, i-d+1) = c_out;
        end       
    end
end