% Generate Smooth Seeds

% (2 eigenvalues + 3 eigenvector components)*(2 terms)*(4 modes)
xis_all = randn([10,4,4,10000]);

save ../smooth_seeds.mat xis_all

xis_all = xis_all(:,:,:,9501:end);
save ../smooth_seeds_test_only.mat xis_all