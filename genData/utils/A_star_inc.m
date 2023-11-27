function a_mat = A_star_inc(node_coords,xis,a11,a12,a22,a33,a34,a44)
    x = node_coords(1)-0.5;
    y = node_coords(2)-0.5;
    K = 5; 
    ks = linspace(1,K,K);
    [theta, rho] = cart2pol(x,y);
    rho_cutoff = 0.25+ xis'.*0.04*sin(ks'*theta);
    
    if rho < rho_cutoff
%         a_mat = [10,3;3,10]; 
        a_mat = [a11,a12;a12,a22];
    else
%         a_mat = [1,0.5;0.5,1
        a_mat = [a33,a34;a34,a44];
    end
    
end
