function s = distddinit(x, y, precision, max_iter, alphafun, adjvec)
    % initialize a distributed dual descent instance. Returns a state
    % struct that needs to be passed into each step.
    s.x = x;
    s.y = y;
    s.precision = precision;
    s.max_iter = max_iter;
    s.alphafun = alphafun;
    s.adjvec = adjvec;
    
    [s.d, s.n] = size(s.x);
    s.x = [ones(1, s.n); s.x];
    s.d = s.d + 1;
    s.theta = zeros(s.d, 1);
    s.llprev = 1e9;
    s.lldelta = 1;
    s.theta_hist = s.theta;
    s.llhist = 0;
    s.iter = 1;
    s.daz = zeros(s.d, 1);
end