function plot_solutions(filename_info,Avals,interp_chi1,interp_chi2)

    %%% Example Usage: 
    % load data/vor_data/data_3.mat
    % plot_solutions('data/vor_data/data_info.mat',Avals,chi1_interp,chi2_interp)
    addpath(genpath('..'));
    load(filename_info);
    
    grid_edge = length(x_grid);

    subplot(1,5,1)
    imagesc(x_grid,y_grid,reshape(Avals(1,1,:),grid_edge,grid_edge))
    title('A')
    colorbar;
    subplot(1,5,2)
    
    
    
    imagesc(x_grid,y_grid,reshape(interp_chi1,grid_edge,grid_edge))
    title('chi1')
    colorbar;

    % Compute div(A)
    %%%% Note: this plot comparison of the LHS and RHS ignores periodic
    %%%% boundaries and is just a sanity check of correctness
    [X,Y] = meshgrid(x_grid,y_grid);

    divA1 = divergence(X,Y,squeeze(reshape(Avals(1,1,:),1,grid_edge,grid_edge)),squeeze(reshape(Avals(1,2,:),1,grid_edge,grid_edge)));
    divA2 = divergence(X,Y,squeeze(reshape(Avals(2,1,:),1,grid_edge,grid_edge)),squeeze(reshape(Avals(2,2,:),1,grid_edge,grid_edge)));

    % Compute LHS
    chi1 = reshape(interp_chi1,grid_edge,grid_edge);
    chi2 = reshape(interp_chi2,grid_edge,grid_edge);
    [gradchi1x, gradchi1y] = gradient(chi1,1/grid_edge);
    gradchi1 = reshape(cat(3,gradchi1x,gradchi1y),grid_edge^2,2)';
    [gradchi2x,gradchi2y] = gradient(chi2,1/grid_edge);
    gradchi2 = reshape(cat(3,gradchi2x,gradchi2y),grid_edge^2,2)';
    Ashift = permute(Avals,[3,1,2]);
    achi1prod = cell2mat(arrayfun(@(i) squeeze(Ashift(i,:,:))*gradchi1(:,i),1:grid_edge^2,'UniformOutput',false));
    achi2prod = cell2mat(arrayfun(@(i) squeeze(Ashift(i,:,:))*gradchi2(:,i),1:grid_edge^2,'UniformOutput',false));
    
    
    LHS1 = -divergence(X,Y,reshape(achi1prod(1,:),grid_edge,grid_edge),reshape(achi1prod(2,:),grid_edge,grid_edge));
    LHS2 = -divergence(X,Y,reshape(achi2prod(1,:),grid_edge,grid_edge),reshape(achi2prod(2,:),grid_edge,grid_edge));
    
    LHS = (LHS1.^2 + LHS2.^2).^(1/2);
    RHS = (divA1.^2+divA2.^2).^(1/2);
    
    subplot(1,5,3)
    imagesc(x_grid,y_grid,reshape(LHS,grid_edge,grid_edge))
    title('LHS')
    colorbar;
    
    subplot(1,5,4)
    imagesc(x_grid,y_grid,reshape(RHS,grid_edge,grid_edge))
    title('RHS')
    colorbar;
    
    
    subplot(1,5,5)
    imagesc(x_grid,y_grid,reshape(((divA1-LHS1).^2 + (divA2-LHS2).^2).^(1/2),grid_edge,grid_edge));
    title('Error')
    colorbar;
end