function Amats = smooth_Amats(n_data,centers,seeds_file)
    addpath(genpath('..'));
    load(seeds_file);
    
    xis = xis_all(:,:,:,n_data);

    Amats = A_smooth(centers,squeeze(xis));
    

end