function Amat = star_fullfieldA(n_data,X,Y,grid_edge,seeds_file)

addpath(genpath('..'));
load(seeds_file)

N_nodes = grid_edge^2;

params = A_params_all(:,n_data);
xis = xis_all(:,n_data);

    A_cells = arrayfun(@(i) A_star_inc([X(i),Y(i)],xis,params(1),params(2),params(3),params(4),params(5),params(6)),1:N_nodes,'UniformOutput',false); % (dim(A) x N_elem)
    A_mat = cell2mat(A_cells);
    Amat = reshape(A_mat,2,2,N_nodes); % A dim 1 x N_data x A dim 2 x N_nodes
    
end
