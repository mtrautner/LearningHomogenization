function Amat = smooth_fullfieldA(n_data,X,Y,grid_edge,seeds_file)
% filename has data info
    addpath(genpath('..'));
    load(seeds_file)

    xis1n = xis1(:,:,n_data);
    xis2n = xis2(:,:,n_data);
    xis3n = xis3(:,:,n_data);
    xis4n = xis4(:,:,n_data);
N_nodes = grid_edge^2;


K = 4;
k1s = linspace(1,K,K);
k2s = linspace(1,K,K);


f_sum1 = zeros(length(X),1);
f_sum2 = zeros(length(X),1);

for k1 = k1s
    for k2 = k2s
        f_sum1= f_sum1 + xis1n(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis2n(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
        f_sum2= f_sum2 + xis3n(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis4n(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
    end
end

abs_f1 = abs(f_sum1);
abs_f2 = abs(f_sum1);
max_f1 = max(abs_f1(:));
max_f2 = max(abs_f2(:));

f_sum1 = f_sum1/(max_f1);
f_sum2 = f_sum2/(max_f2);
eig1 = exp(f_sum1);
eig2 = exp(f_sum2);

A_cells = arrayfun(@(i) [eig1(i), 0; 0, eig2(i)], 1:length(X),'UniformOutput',false);

A_mat = cell2mat(A_cells);
Amat = reshape(A_mat,2,2,N_nodes); % A dim 1 x N_data x A dim 2 x N_nodes


end
