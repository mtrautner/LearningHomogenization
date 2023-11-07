function Amat = vor_fullfieldA(n_data,X,Y,grid_edge,seeds_file)

addpath(genpath('..'));
load(seeds_file)

N_nodes = grid_edge^2;

point_N = vorCell{n_data,1};
points = vorCell{n_data,2};
a11s = vorCell{n_data,3};
a12s = vorCell{n_data,4};
a22s = vorCell{n_data,5};

all_points = zeros(point_N*9,2);
    counter = 0;
    for i = [0,1,-1]
        for j = [0,1,-1]
            xs = points(:,1) + i;
            ys = points(:,2) + j;
            all_points(counter*point_N + 1:counter*point_N+point_N,:) = [xs,ys];
            counter = counter + 1;
        end
    end

pcas_grid = dsearchn(all_points,[X';Y']');
pcas_grid = mod(pcas_grid,point_N) + 1;

  
    
    A_grid = arrayfun(@(i) [a11s(pcas_grid(i)), a12s(pcas_grid(i)); a12s(pcas_grid(i)), a22s(pcas_grid(i))], 1:length(X),'UniformOutput',false); % (dim(A) x N_elem)

    A_grid = cell2mat(A_grid);

    Amat = reshape(A_grid,2,2,length(X));

end
