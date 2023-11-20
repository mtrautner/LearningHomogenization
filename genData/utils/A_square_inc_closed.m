function a_mat = A_square_inc_closed(node_coords,r,a11,a12,a22,a33,a34,a44)
    x = node_coords(1)-0.5;
    y = node_coords(2)-0.5;
    eps = 1E-13;
    if abs(x) <= r + eps && abs(y) <= r+eps
%         a_mat = [10,3;3,10]; 
        a_mat = [a11,a12;a12,a22];
    else
%         a_mat = [1,0.5;0.5,1
        a_mat = [a33,a34;a34,a44];
    end
    
end