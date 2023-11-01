function Amat = sq_fullfieldA(n_data,X,Y,grid_edge,seeds_file)

addpath(genpath('..'));
load(seeds_file)

N_nodes = grid_edge^2;

    vals = params_all(:,n_data);
    [r,a11,a12,a22,a33,a34,a44] = deal(vals(1),vals(2),vals(3),vals(4),vals(5),vals(6),vals(7));

    A_cells = arrayfun(@(i) A_square_inc([X(i),Y(i)],r,a11,a12,a22,a33,a34,a44),1:N_nodes,'UniformOutput',false); % (dim(A) x N_elem)
%     A_cells_vert = vertcat(A_cells{:});
    A_mat = cell2mat(A_cells);
    Amat = reshape(A_mat,2,2,N_nodes); % A dim 1 x N_data x A dim 2 x N_nodes
    

end

