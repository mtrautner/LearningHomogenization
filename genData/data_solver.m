function data_solver(seeds_file,A_mats_func,A_fullfield_func,h,N_data,grid_edge,save_dir)

% Define the domain and discretize into elements
% Basic method adapted from https://www.particleincell.com/2012/matlab-fem/
% load data_seeds/smooth_seeds.mat
load(seeds_file)
addpath(genpath('utils'));

%%%%%% Input Parameters %%%%%%%%%%%% Below are parameter descriptions as
%%%%%% well as recommended values for h and grid_edge
% seeds_file: set random seeds for data generation first.
% A_mats func: for microstructure of choice, pass in function to construct
% Amats
% A_fullfield_func: for microstructure of choice, pass in function to
% construct the fullfield of A on a grid from the seeds
% h = 0.005; % Adjust this to change number of elements in the solver
% N_data = 1; % Adjust this to change the number of data to generate solutions to
% grid_edge = 128; % Adjust this to change interpolation grid size 
% filename_info = ['data/smooth_data/data_info.mat']; % Adjust where the data info is saved
% filename_data = 'data/smooth_data/smooth_data_'; % prefix for where data is saved

%%%%%% Note about plots: 

%%%%%% the LHS and RHS and Error computations do not consider
%%%%%% periodic boundaries. The plots are just a sanity check that the
%%%%%% solver is working correctly and a tool to visualize A and \chi_1

% Coordinates
lowerLeft  = [0   ,0   ];
lowerRight = [1,0];
upperRight = [1,1];
upperLeft =  [0,1];
% Geometry matrix
S = [3,4, lowerLeft(1), lowerRight(1), upperRight(1), upperLeft(1), lowerLeft(2), lowerRight(2), upperRight(2), upperLeft(2)];                     
gdm = S';
% Names
ns = 'S';
% Set formula 
sf = 'S';

% Invoke decsg
g = decsg(gdm,ns,sf');
model = createpde;
g = geometryFromEdges(model,g);
mesh = generateMesh(model,"Hmax",h,"GeometricOrder", "linear");
[p,e,t] = meshToPet(model.Mesh); % p is node coordinates, t is nodes for each element
p = model.Mesh.Nodes;
t = model.Mesh.Elements;
N_edge = size (p ,2) ;
N_elem = size (t ,2) ;
N_node = size (p ,2) ;

% Identify boundaries
bottom_edge_nodes = findNodes(mesh,"region","Edge",1);
top_edge_nodes = findNodes(mesh,"region","Edge",3);
right_edge_nodes = findNodes(mesh,"region","Edge",2);
left_edge_nodes = findNodes(mesh,"region","Edge",4);
left_edge_nodes = flip(left_edge_nodes); % order bottom to top
top_edge_nodes = flip(top_edge_nodes); % order left to right
bottom_top_pairs = cat(1,bottom_edge_nodes,top_edge_nodes);
left_right_pairs = cat(1,left_edge_nodes, right_edge_nodes);
b_length = length(bottom_top_pairs);

ind_vec1 = [1,0]'; 
ind_vec2 = [0,1]';


% Make coarser grid
x_grid = linspace(0,1,grid_edge);
y_grid = linspace(0,1,grid_edge);
[X,Y] = meshgrid(x_grid,y_grid);
X = reshape(X,grid_edge^2,1);
Y = reshape(Y,grid_edge^2,1);
filename_info = strcat(save_dir,'data_info.mat');
save(filename_info, 'x_grid', 'y_grid', 'p', 't')
parpool(8) %%%% This enforces running in parallel. To make not parallel, comment out and change the parfor below to "for"

parfor n_data = 1:N_data

    F1 = zeros(N_node,1); % RHS
    F2 = zeros(N_node,1);
%     xis1n = xis1(:,:,n_data);
%     xis2n = xis2(:,:,n_data);
%     xis3n = xis3(:,:,n_data);
%     xis4n = xis4(:,:,n_data);
    

    node_ind_flat = reshape(t,1,[]); % (1 x 3*N_elem)
    node_coords_all = p(:,node_ind_flat)'; % (3*N_elem x 2)
    node_coords_reshaped = reshape(node_coords_all,3,N_elem,2); % (3 x N_elem x 2)
    node_coords_reshaped = permute(node_coords_reshaped,[2,1,3]); % (N_elem x 3 x 2)
    Pe_all = cat(3,ones(N_elem,3),node_coords_reshaped); % (N_elem x 3 x 3) 
    Areas = arrayfun(@(i) abs(det(squeeze(Pe_all(i,:,:))))/2, 1:N_elem); % (1 x N_elem)
    centers = mean(node_coords_reshaped,2); % (N_elem x 2)
%     A_mats = A_smooth(centers,xis1n,xis2n,xis3n,xis4n);
    A_mats = A_mats_func(n_data,centers,seeds_file);

    C_all = arrayfun(@(i) inv(squeeze(Pe_all(i,:,:))), 1:N_elem,'UniformOutput',false); % (1 x N_elem) cell array: ea cell is 3x3 matrix
    C_all_array = cell2mat(C_all); % (3 x 3*N_elem) array
    C_all_array1 = reshape(C_all_array,3,3,N_elem); % (3 x 3 x N_elem);
    grad = C_all_array1(2:3,:,:); % (2 x 3 x N_elem) R2 gradients for each of 3 node funcs oer each element
    
    % Computing K and F values for each element
    Kes = arrayfun(@(i) (Areas(i)*A_mats{i}*grad(:,:,i))'*grad(:,:,i),1:N_elem,'UniformOutput',false);
    Fes1 = arrayfun(@(i) -Areas(i)*(A_mats{i}*ind_vec1)'*grad(:,:,i), 1:N_elem, 'UniformOutput', false);
    Fes2 = arrayfun(@(i) -Areas(i)*(A_mats{i}*ind_vec2)'*grad(:,:,i), 1:N_elem, 'UniformOutput', false);

    
    % Assembling stiffness and forcing matrices
    to_be_K = zeros(9,3,N_elem);

    for elem = 1:N_elem
        nodes = t(:,elem);
        Ke = Kes{elem};
        Fe1 = Fes1{elem};
        Fe2 = Fes2{elem};

        [node_mesh1, node_mesh2] = meshgrid(nodes,nodes);
        node_mesh2 = cat(2,node_mesh1',node_mesh2');
        node_mesh3 = reshape(node_mesh2,[],2);

        Ke_flat = reshape(Ke,9,1);
        to_be_K(:,1:2,elem) = node_mesh3;
        to_be_K(:,3,elem) = Ke_flat;

        F1(nodes) = F1(nodes) + Fe1'; 
        F2(nodes) = F2(nodes) + Fe2';
    end

    to_be_K_reshaped = reshape(permute(to_be_K,[1,3,2]),9*N_elem,3);
    i_s = to_be_K_reshaped(:,1);
    j_s = to_be_K_reshaped(:,2);
    v_s = to_be_K_reshaped(:,3);

    K = sparse(i_s,j_s,v_s);


    % % Dirichlet BC
    % K(b,:) = 0; K(:,b) = 0; F(b) = 0; 
    % K(b,b) = speye(length(b),length(b)); % put I into boundary submatrix of K

    % % Periodic BC


    % Add values
    K(bottom_edge_nodes,:) = K(bottom_edge_nodes,:) + K(top_edge_nodes,:);
    K(:,bottom_edge_nodes) = K(:,bottom_edge_nodes) + K(:,top_edge_nodes);
    K(left_edge_nodes,:) = K(left_edge_nodes,:) + K(right_edge_nodes,:);
    K(:,left_edge_nodes) = K(:,left_edge_nodes) + K(:,right_edge_nodes);
    F1(bottom_edge_nodes) = F1(bottom_edge_nodes) + F1(top_edge_nodes);
    F1(left_edge_nodes) = F1(left_edge_nodes) + F1(right_edge_nodes);

    F2(bottom_edge_nodes) = F2(bottom_edge_nodes) + F2(top_edge_nodes);
    F2(left_edge_nodes) = F2(left_edge_nodes) + F2(right_edge_nodes);


    % Clear entries
    K(right_edge_nodes,:) = 0;
    K(top_edge_nodes,:) = 0;
    F1(right_edge_nodes) = 0;
    F1(top_edge_nodes) = 0;
    F2(right_edge_nodes) = 0;
    F2(top_edge_nodes) = 0;
    K(:,right_edge_nodes) = 0;
    K(:,top_edge_nodes) = 0;

    top_right_corner = right_edge_nodes(end);
    bottom_right_corner = right_edge_nodes(1);
    bottom_left_corner = bottom_edge_nodes(1);
    % Add equality conditions
    for i = 1:b_length
        K(right_edge_nodes(i),left_edge_nodes(i)) = -1;
        K(right_edge_nodes(i), right_edge_nodes(i)) = 1;

        K(top_edge_nodes(i), bottom_edge_nodes(i)) = -1;
        K(top_edge_nodes(i), top_edge_nodes(i)) = 1;
    end

    K(top_right_corner,bottom_right_corner) = 0;


    % Bottom left corner is unspecified and has no map to it: need to specify
    % it

    K(bottom_left_corner,:) = 0;
    K(bottom_left_corner,bottom_left_corner) = 1;
    F1(bottom_left_corner) = 0;
    F2(bottom_left_corner) = 0;
    

    % % % Solve for U
    
    chi1 = K\F1;
    chi2 = K\F2;
%     solve_error1 = max(abs(K*chi1 - F1))
%     solve_error2 = max(abs(K*chi2 - F2))
% 

    
    % Interpolate finer grid solution to coarser grid
    
    V_chi1 = scatteredInterpolant(p(1,:)',p(2,:)', chi1);
    V_chi2 = scatteredInterpolant(p(1,:)',p(2,:)', chi2);
    
    interp_chi1 = V_chi1(X,Y);
    interp_chi2 = V_chi2(X,Y);
    
    filename = strcat(save_dir,'data_', num2str(n_data,'%5d'),'.mat');
    
    % Get fullfield A values on the interpolation grid
    Avals = feval(A_fullfield_func,n_data,X,Y,grid_edge,seeds_file);
    parsave(filename, Avals, interp_chi1, interp_chi2);
    

end


end
