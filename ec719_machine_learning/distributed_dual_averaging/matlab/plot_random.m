function plot_random(x, y, A)
    % plot a random graph generated by gen_random()
    scatter(x, y);
    hold on;
    l = max([x y]);
    for i = 1:size(A, 1)
        for j=i:size(A, 1)
            if A(i,j) > 0 
                plot([x(i) x(j)], [y(i) y(j)]);    
            end
        end
    end 
    axis([0 l 0 l]);
end