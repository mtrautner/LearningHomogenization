function A_mats = sq_Amats_closed(n_data,centers,seeds_file)
    addpath(genpath('..'));
    load(seeds_file);
    
    N_elem = length(centers);
    vals = params_all(:,n_data);
    [r,a11,a12,a22,a33,a34,a44] = deal(vals(1),vals(2),vals(3),vals(4),vals(5),vals(6),vals(7));
    A_mats = arrayfun(@(i) A_square_inc_closed(centers(i,:),r,a11,a12,a22,a33,a34,a44),1:N_elem,'UniformOutput',false); % (dim(A) x N_elem)

    
end