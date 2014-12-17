function A = gen_cycle(n)
% generate a doubly stochastic matrix for a singly linked cycle of size n
    A = circshift(eye(n), [1 0]) ./ 3 + ...
        circshift(eye(n), [-1 0]) ./ 3 + eye(n) ./ 3;
end