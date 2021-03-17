function [stress_cart, property, DD, lambda] = stress_interp_hex(Cart_data,Elements,Nodes,stress_map,coord)

%% Retrieve data values

center    = [coord(1),  coord(2)];
circle_r  = coord(3);
left_bot  = [coord(4),  coord(5)];
right_bot = [coord(6),  coord(7)];
left_top  = [coord(8),  coord(9)];
right_top = [coord(10), coord(11)];

%% Reinterpolating stress onto cart_map

L           = length(Cart_data(:,1));
stress_cart = zeros(L,1);
property    = zeros(L,1);
DD          = zeros(L,3,3);
lambda      = zeros(L,4);

for i = 1:L    
    
    node = [Cart_data(i,1),Cart_data(i,2)];
    
    for j = 1:length(Elements(:,1))
        
        ele_1 = Elements(j,2);
        ele_2 = Elements(j,3);
        ele_3 = Elements(j,4);
        
        node_1 = [Nodes(ele_1,1),Nodes(ele_1,2)];
        node_2 = [Nodes(ele_2,1),Nodes(ele_2,2)];
        node_3 = [Nodes(ele_3,1),Nodes(ele_3,2)];
        
        centroid = 1/3 * (node_1 + node_2 + node_3);
        
        lambda_1 = ((node_2(2)-node_3(2))*(node(1)-node_3(1)) + (node_3(1)-node_2(1))*(node(2)-node_3(2))) / ((node_2(2)-node_3(2))*(node_1(1)-node_3(1)) + (node_3(1)-node_2(1))*(node_1(2)-node_3(2)));
   
        lambda_2 = ((node_3(2)-node_1(2))*(node(1)-node_3(1)) + (node_1(1)-node_3(1))*(node(2)-node_3(2))) / ((node_2(2)-node_3(2))*(node_1(1)-node_3(1)) + (node_3(1)-node_2(1))*(node_1(2)-node_3(2)));
        
        lambda_3 = 1 - lambda_1 - lambda_2;
        
        % Clear out the extremely low values to zero to avoid confusion 
        % This part of code is extremely necessary to avoid errors
        if abs(lambda_1) < 1e-10
            lambda_1 = 0;
            
        elseif abs(lambda_2) < 1e-10
            lambda_2 = 0;
            
        elseif abs(lambda_3) < 1e-10    
            lambda_3 = 0;
        
        end
        
        % Judging the elelment location and determine interpolation values
        if (lambda_1 >= 0) && (lambda_2 >= 0) && (lambda_3 >= 0)
            
            stress_cart(i,1) = lambda_1*stress_map(ele_1) + lambda_2*stress_map(ele_2) + lambda_3*stress_map(ele_3);
            
            if ((norm(centroid-center)<=circle_r))||(norm(centroid-left_bot)<=circle_r)||(norm(centroid-left_top)<=circle_r)||(norm(centroid-right_top)<=circle_r)||(norm(centroid-right_bot)<=circle_r)
                
                property(i,1) = 1;
                
            else
                property(i,1) = 2;
                
            end
            
            lambda(i,1) = lambda_1;
            lambda(i,2) = lambda_2;
            lambda(i,3) = lambda_3;
            lambda(i,4) = j;
            
            break
            
        else
            
            continue
            
        end
                
    end
          
end