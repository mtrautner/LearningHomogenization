function A_mats = star_Amats(n_data,centers,seeds_file)
    addpath(genpath('..'));
    load(seeds_file);
    
    N_elem = length(centers);
    vals = A_params_all(:,n_data);
    xis = xis_all(:,n_data);
    [a11,a12,a22,a33,a34,a44] = deal(vals(1),vals(2),vals(3),vals(4),vals(5),vals(6));
    A_mats = arrayfun(@(i) A_star_inc(centers(i,:),xis,a11,a12,a22,a33,a34,a44),1:N_elem,'UniformOutput',false); % (dim(A) x N_elem)

    

end