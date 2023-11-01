function A_mats = vor_Amats(n_data,centers,seeds_file)
    addpath(genpath('..'));
    load(seeds_file);
    
    N_elem = length(centers);
    
    point_N = vorCell{n_data,1};
    points = vorCell{n_data,2};
    a11s = vorCell{n_data,3};
    
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
    pcas = dsearchn(all_points,squeeze(centers));
    pcas = mod(pcas,point_N) + 1;
    
    
    a22s = a11s; 
    a12s = 0*a11s; 
  
    A_mats = arrayfun(@(i) [a11s(pcas(i)), a12s(pcas(i)); a12s(pcas(i)), a22s(pcas(i))], 1:N_elem,'UniformOutput',false); 
    
end