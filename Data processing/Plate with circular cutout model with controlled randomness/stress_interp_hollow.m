function [stress_cart, DD] = stress_interp_hollow(Cart_data,Elements,Nodes,stress_map)

%% Reshaping Stress Map to a vector
% The stress_map needs to be reshaped to N-by-1 vector
%
%

%% Reinterpolating stress onto cart_map

L = length(Cart_data(:,1));
stress_cart = zeros(L,1);
DD = zeros(L,3,3);

for i = 1:L    
    
    node = [Cart_data(i,1),Cart_data(i,2)];
    
    for j = 1:length(Elements(:,1))
        
        ele_1 = Elements(j,2);
        ele_2 = Elements(j,3);
        ele_3 = Elements(j,4);
        
        node_1 = [Nodes(ele_1,1),Nodes(ele_1,2)];
        node_2 = [Nodes(ele_2,1),Nodes(ele_2,2)];
        node_3 = [Nodes(ele_3,1),Nodes(ele_3,2)];
        
        lambda_1 = ((node_2(2)-node_3(2))*(node(1)-node_3(1)) + (node_3(1)-node_2(1))*(node(2)-node_3(2))) / ((node_2(2)-node_3(2))*(node_1(1)-node_3(1)) + (node_3(1)-node_2(1))*(node_1(2)-node_3(2)));
        lambda_2 = ((node_3(2)-node_1(2))*(node(1)-node_3(1)) + (node_1(1)-node_3(1))*(node(2)-node_3(2))) / ((node_2(2)-node_3(2))*(node_1(1)-node_3(1)) + (node_3(1)-node_2(1))*(node_1(2)-node_3(2)));
        lambda_3 = 1 - lambda_1 - lambda_2;
        
        if (lambda_1 >= 0) && (lambda_2 >= 0) && (lambda_3 >= 0)
            
            stress_cart(i,1) = lambda_1*stress_map(ele_1) + lambda_2*stress_map(ele_2) + lambda_3*stress_map(ele_3);
            
            break
            
        else
            
            continue
            
        end
                
    end
          
end