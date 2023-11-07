% Generate Voronoi data

N_samples = 10000;

N_points_min = 5;
N_points_max = 5;
point_counts = randi([N_points_min, N_points_max],N_samples,1);

mesh_points = [];

vorCell = cell(N_samples, 5);
E = exp(1);

for n = 1:N_samples
    point_N = point_counts(n);
    points = rand(point_N,2);
    dt = delaunayTriangulation(points(:,1),points(:,2));
    
    eig1 = 1/E + (E-1/E)*rand(point_N,1);
    eig2 = 1/E + (E-1/E)*rand(point_N,1);
    a11s = zeros(point_N,1);
    a12s = zeros(point_N,1);
    a22s = zeros(point_N,1);
    for k = 1:point_N
        eigvecs = randn(2);
        eigvecs(2,2) = -eigvecs(1,1)*eigvecs(1,2)/eigvecs(2,1);
        eigvecs = eigvecs./(vecnorm(eigvecs));
        M = diag([eig1(k),eig2(k)]);
        A1 = eigvecs*M*inv(eigvecs);
        a11s(k) = A1(1,1);
        a12s(k) = A1(1,2);
        a22s(k) = A1(2,2);
    end
    vorCell{n,1} = point_N;
    vorCell{n,2} = points;
    vorCell{n,3} = a11s;
    vorCell{n,4} = a12s;
    vorCell{n,5} = a22s;
%     a22s = 0.1 + 10*rand(point_N,1);
%     a12s = 0.1 + (min(a11s,a22s) - 0.011)*rand(1);
    
%     point_cell_assignment = pointLocation(dt,mesh_points);
    
end

save ../vor_seeds.mat vorCell

vorCell = vorCell(9501:10000,:);
save ../vor_seeds_test_only.mat vorCell