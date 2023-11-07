K = 5; 
N_data = 10000; 
% 
ks = linspace(1,K,K);
thetas = linspace(0,2*pi,200);
for n_data = 1:N_data
    xis = 2*rand(K,1)-1;
    phiks = 0.04*sin(ks'*thetas);
    phi_sum = 0.25+ xis'*phiks;
end

polarplot(thetas,phi_sum)

xis_all = zeros(5,N_data);
A_params_all = zeros(6,N_data);
r = 0.02 + (0.48-0.02)*rand(1); % radius of square inclusion between (0.02, 0.48)
a11 = 0; a12= 0; a22= 0; a33 = 0; a34 = 0; a44 = 0; 
E = exp(1);
for n_data = 1:N_data
    xis = 2*rand(K,1)-1;
    for i = [1,2]
        eig1 = 1/E+ (E-1/E)*rand(1);
        eig2 = 1/E + (E-1/E)*rand(1);
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
    xis_all(:,n_data) = xis;
    A_params_all(:,n_data) = [a11,a12,a22,a33,a34,a44];
end

save ../star_inc_seeds.mat xis_all A_params_all

xis_all = xis_all(:,9501:end);
A_params_all = A_params_all(:,9501:end);
save ../star_inc_seeds_test_only.mat xis_all A_params_all