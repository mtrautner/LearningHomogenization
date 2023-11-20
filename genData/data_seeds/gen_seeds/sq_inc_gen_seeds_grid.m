N_data = 500;
params_all = zeros(7,N_data);
grid_edge = 128;
x_grid = linspace(0,1,grid_edge);
allowed_ticks = x_grid(10:end-10);

a11 = 0; a12= 0; a22= 0; a33 = 0; a34 = 0; a44 = 0; 
E = exp(1);
for n_data = 1:N_data
    tick = randsample(allowed_ticks,1); % radius of square inclusion between (0.02, 0.48)
    r = abs(tick-0.5);
    for i = [1,2]
        eig1 = 1/E+ (E-1/E)*rand(1);
        eig2 = 1/E+ (E-1/E)*rand(1);
        eigvecs = randn(2);
        eigvecs(2,2) = -eigvecs(1,1)*eigvecs(1,2)/eigvecs(2,1);
        eigvecs = eigvecs./(vecnorm(eigvecs));
        M = diag([eig1,eig2]);
        A1 = eigvecs*M*inv(eigvecs);
        if i == 1
            a11 = A1(1,1);
            a12 = A1(1,2);
            a22 = A1(2,2);
        end
        if i == 2
            a33 = A1(1,1);
            a34 = A1(1,2);
            a44 = A1(2,2);
        end
    end
    params_all(:,n_data) = [r,a11,a12,a22,a33,a34,a44];
end

save ../sq_inc_seeds_grid.mat params_all
