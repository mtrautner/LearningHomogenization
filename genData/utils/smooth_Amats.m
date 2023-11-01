function Amats = smooth_Amats(n_data,centers,seeds_file)
    addpath(genpath('..'));
    load(seeds_file);
    
    xis1n = xis1(:,:,n_data);
    xis2n = xis2(:,:,n_data);
    xis3n = xis3(:,:,n_data);
    xis4n = xis4(:,:,n_data);
    
    Amats = A_smooth(centers,xis1n,xis2n,xis3n,xis4n);
    

end