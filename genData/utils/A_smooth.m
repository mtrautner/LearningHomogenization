function A_mats = A_smooth(centers, xis_all)

    K = 4;
    k1s = linspace(1,K,K);
    k2s = linspace(1,K,K);
    
    xis1 = squeeze(xis_all(1,:,:));
    xis2 = squeeze(xis_all(2,:,:));
    xis3 = squeeze(xis_all(3,:,:));
    xis4 = squeeze(xis_all(4,:,:));
    xis5 = squeeze(xis_all(5,:,:));
    xis6 = squeeze(xis_all(6,:,:));
    xis7 = squeeze(xis_all(7,:,:));
    xis8 = squeeze(xis_all(8,:,:));
    xis9 = squeeze(xis_all(9,:,:));
    xis10 = squeeze(xis_all(10,:,:));

    X = centers(:,1);
    Y = centers(:,2);
    
%     f_sum1 = zeros(grid_edge,grid_edge);
%     f_sum2 = zeros(grid_edge,grid_edge);
% Two eigenvalues and three eigenvector components
    eig1 = zeros(length(centers),1);
    eig2 = zeros(length(centers),1);
    vec1 = zeros(length(centers),1);
    vec2 = zeros(length(centers),1);
    vec3 = zeros(length(centers),1);
    
    for k1 = k1s
        for k2 = k2s
            eig1= eig1 + xis1(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis2(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
            eig2= eig2 + xis3(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis4(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
            vec1= vec1 + xis5(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis6(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
            vec2= vec2 + xis7(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis8(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
            vec3= vec3 + xis9(k1,k2)*sin(2*pi*k1*X).*cos(2*pi*k2*Y) + xis10(k1,k2)*cos(2*pi*k1*X).*sin(2*pi*k2*Y);
        end
    end

    abs_eig1 = abs(eig1);
    abs_eig2 = abs(eig2);
    abs_vec1 = abs(vec1);
    abs_vec2 = abs(vec2);
    abs_vec3 = abs(vec3);
    max_eig1 = max(abs_eig1(:));
    max_eig2 = max(abs_eig2(:));
    max_vec1 = max(vec1(:));
    max_vec2 = max(vec2(:));
    max_vec3 = max(vec3(:));

    f_sum1 = eig1/(max_eig1);
    f_sum2 = eig2/(max_eig2);
    v_sum1 = vec1/(max_vec1);
    v_sum2 = vec2/(max_vec2);
    v_sum3 = vec3/(max_vec3);
    eig1 = exp(f_sum1);
    eig2 = exp(f_sum2);
    vec1 = exp(v_sum1);
    vec2 = exp(v_sum2);
    vec3 = exp(v_sum3);
    
    
    
    eigvecs = arrayfun(@(i) [vec1(i), vec3(i); vec2(i) -vec1(i)*vec3(i)/vec2(i)], 1:length(centers),'UniformOutput',false);
    eigvecs_scaled = arrayfun(@(i) eigvecs{i}./(vecnorm(eigvecs{i})), 1:length(centers),'UniformOutput',false);
    A_eigs = arrayfun(@(i) [eig1(i), 0; 0, eig2(i)], 1:length(centers),'UniformOutput',false);
    A_mats = arrayfun(@(i) eigvecs_scaled{i}*A_eigs{i}*inv(eigvecs_scaled{i}),1:length(centers),'UniformOutput',false);
end