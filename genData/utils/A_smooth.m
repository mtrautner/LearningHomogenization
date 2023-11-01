function A_mats = get_smooth_A_mats(centers, xis1, xis2, xis3, xis4)

    K = 4;
    k1s = linspace(1,K,K);
    k2s = linspace(1,K,K);

%     x_grid = linspace(0,1,grid_edge);
%     y_grid = linspace(0,1,grid_edge);
%     [X,Y] = meshgrid(x_grid,y_grid);

    X = centers(:,1);
    Y = centers(:,2);

%     f_sum1 = zeros(grid_edge,grid_edge);
%     f_sum2 = zeros(grid_edge,grid_edge);
    f_sum1 = zeros(length(centers),1);
    f_sum2 = zeros(length(centers),1);
    
    for k1 = k1s
        for k2 = k2s
            f_sum1= f_sum1 + xis1(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis2(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
            f_sum2= f_sum2 + xis3(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis4(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
        end
    end

    abs_f1 = abs(f_sum1);
    abs_f2 = abs(f_sum1);
    max_f1 = max(abs_f1(:));
    max_f2 = max(abs_f2(:));

    f_sum1 = f_sum1/(max_f1);
    f_sum2 = f_sum2/(max_f2);
    eig1 = exp(f_sum1);
    eig2 = exp(f_sum2);
    
    A_mats = arrayfun(@(i) [eig1(i), 0; 0, eig2(i)], 1:length(centers),'UniformOutput',false);

end