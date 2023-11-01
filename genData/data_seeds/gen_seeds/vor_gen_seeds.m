% Generate Voronoi data

N_samples = 10000;

N_points_min = 5;
N_points_max = 5;
point_counts = randi([N_points_min, N_points_max],N_samples,1);

mesh_points = [];

vorCell = cell(N_samples, 3);

for n = 1:N_samples
    point_N = point_counts(n);
    points = rand(point_N,2);
    dt = delaunayTriangulation(points(:,1),points(:,2));
    
    
    a11s = 0.1 + 10*rand(point_N,1);
    
    vorCell{n,1} = point_N;
    vorCell{n,2} = points;
    vorCell{n,3} = a11s;
%     a22s = 0.1 + 10*rand(point_N,1);
%     a12s = 0.1 + (min(a11s,a22s) - 0.011)*rand(1);
    
%     point_cell_assignment = pointLocation(dt,mesh_points);
    
end

save ../vor_seeds.mat vorCell