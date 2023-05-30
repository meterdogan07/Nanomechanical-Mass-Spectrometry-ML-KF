function out = selective_mean(A) % only considers non negative entries
    [m, n] = size(A);
    out = zeros(m, 1);

    for i=1:m
        to_mean = 0;
        for j=1:n
            if(A(i,j) >= 0)
                to_mean = to_mean + 1;
                out(i) = out(i) + A(i, j);
            end
        end
        out(i) = out(i)/to_mean;
    end
end
