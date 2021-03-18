% clear all
close all
clc

%% Stress criteria

% stress_disp = 5;

%% Defining parameters

rectangle_x = 0.01;
rectangle_y = 0.01;

n_x = 79;
n_y = 79;
M = [n_y+1 n_x+1];

data_set = 650;

% E_f = 8000;
% E_c = 3200;

%% Contour Postprocessing

folder_path = 'C:\Temp_Abaqus\micro_meter_model\random_hole_hollow_four\';

for iii = 1:data_set
    
    %% Parameters configuration
    
    idx = num2str(iii+1350);
    path_c = strcat(folder_path,'composite_cir_cen_',idx,'.dat');
    center = load(path_c);
    
    circle_x1 = center(1);
    circle_y1 = center(2);
    circle_x2 = center(3);
    circle_y2 = center(4);
    circle_x3 = center(5);
    circle_y3 = center(6);
    circle_x4 = center(7);
    circle_y4 = center(8);
    circle_r  = center(9);
    
    dw = circle_r * 2 / rectangle_x;
    
%     K = 3 - 3.14*dw + 3.667*dw^2 - 1.527*dw^3; 

    K = 1;
    
    %% Generating Cartesian Map and write it out

    Cart_data = Cartesian_Map(rectangle_x, rectangle_y, n_x, n_y);

    fileName = fopen('Cart_Map.dat','w');

    fprintf(fileName,'%f %f\n', Cart_data);

    fclose(fileName);

    %% Cartesian Map Distance to origin
    
    clear Cart_dist1
    clear Cart_dist2
    clear Cart_dist3
    clear Cart_dist4
    
    L_Cart = length(Cart_data(:,1));
    Cart_dist1 = zeros(L_Cart,1);
    Cart_dist2 = zeros(L_Cart,1);
    Cart_dist3 = zeros(L_Cart,1);
    Cart_dist4 = zeros(L_Cart,1);
    
    for ii = 1:1:L_Cart

        Cart_dist1(ii,1) = distance(circle_x1,circle_y1,Cart_data(ii,1),Cart_data(ii,2));
        Cart_dist2(ii,1) = distance(circle_x2,circle_y2,Cart_data(ii,1),Cart_data(ii,2));
        Cart_dist3(ii,1) = distance(circle_x3,circle_y3,Cart_data(ii,1),Cart_data(ii,2));
        Cart_dist4(ii,1) = distance(circle_x4,circle_y4,Cart_data(ii,1),Cart_data(ii,2));

    end     
    
    % Generate stress indicator contour
    
        
    %% Interpolate stress data onto Cartesian Map

    % U1 U2 S11 S22 S33 S12
    stress_file = strcat(folder_path,'loadDisp_New_Master_von_',idx,'.dat');
    stress = load(stress_file);
    % [L_s,~] = size(stress);
    
    node_file = strcat(folder_path,'node_coord_',idx,'.dat');
    aba_nodes = load(node_file);

    aba_nodes = aba_nodes(:,1:2);
    stress_map = zeros(length(stress(:,3)),1);
%     
    element_file = strcat(folder_path,'element_indice_',idx,'.dat');
    aba_elements = load(element_file);

    for mm = 1:length(stress(:,3))
        
        stress_map(mm,1) = sqrt(stress(mm,3)^2+stress(mm,4)^2-stress(mm,3)*stress(mm,4)+3*stress(mm,6)^2);
        
    end
    
    % stress_map = stress(:,3);
    % stress_map = stress(L_s/2+1:end,3);

    [stress_cart,D] = stress_interp_hollow(Cart_data,aba_elements,aba_nodes,stress_map);

    pp1 = 0:rectangle_x/n_x:rectangle_x;
    pp2 = 0:rectangle_y/n_y:rectangle_y;

    [x,y] = meshgrid(pp1,pp2);

%    Remove areas without any nodes in geometry

    for f = 1:length(Cart_dist1(:,1))

        if (Cart_dist1(f,1)<circle_r) || (Cart_dist2(f,1)<circle_r) || (Cart_dist3(f,1)<circle_r) || (Cart_dist4(f,1)<circle_r)

            stress_cart(f) = 0;

        end
    end
        
    
    % Plot stress contour

%     figure(4*iii-3)
%     [X,Y,Z] = griddata(Cart_data(:,1),Cart_data(:,2),stress_cart,x,y,'cubic');
%     contourf(X,Y,Z,10)
%     title('interpolated contour')
%     colorbar
%     shading interp
% 
%     figure(4*iii-2)
%     [X_a,Y_a,Z_a] = griddata(aba_nodes(:,1),aba_nodes(:,2),stress_map,x,y,'linear');
%     contourf(X_a,Y_a,Z_a,10)
%     title('original abaqus contour')
%     colorbar
%     shading interp

    % write out the stress_cart matrix
    
    stress_cart_file = strcat(folder_path,'Composite_uniform_Stress_Cart_',idx,'.dat');
    
    fid = fopen(stress_cart_file,'w');
    
    stress_matrix = [Cart_data stress_cart];
    stress_matrix = stress_matrix.';
    
    fprintf(fid,'%d %d %d\n',stress_matrix);
    
    fclose(fid);
    
    %% Generating Signed Distance Function

    Tmp = zeros(n_y+1,n_x+1);

%     figure(4*iii-1)
% 
%     surf(reshape(Cart_data(:,1), M),reshape(Cart_data(:,2), M), Tmp, 'facecolor',[1 1 1]');


    %%% If the node belongs to SDF_1, weight = 0;
    %%% If the node belongs to SDF_2, weight = E1 / E2;
    %%% If the node belongs to SDF_3, weight = 1;
    
    % pick up the elliptical area near stress concentration location
    % the coefficient should be determined by some percentage of stress
    % line
    
    % First determining the radius of circle
     
%     ratio = [1, 0.8, 0.6, 0.4, 0.2];

    ratio = 0.5;
    
    L_K_r = length(ratio);
     
    K_r = zeros(L_K_r, 1);
    radius = zeros(L_K_r, 1);
 
%     for ff = 1:L_K_r
%          
%         [K_r(ff), radius(ff)] = stress_radius(circle_r, rectangle_x, ratio(ff), K);
%          
%     end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear Cart_dist
    Cart_dist = zeros(L_Cart,1);
    
    for jj = 1:length(Cart_dist(:,1))
        
%         cir = circle_dist(Cart_data(jj,1),Cart_data(jj,2),circle_x,circle_y,r);
        
        if (Cart_dist1(jj,1)<circle_r) || (Cart_dist2(jj,1)<circle_r) || (Cart_dist3(jj,1)<circle_r) || (Cart_dist4(jj,1)<circle_r)

            
            Cart_dist(jj,1) = 0;
            
        elseif (Cart_dist1(jj,1)<circle_r) || (Cart_dist2(jj,1)<circle_r) || (Cart_dist3(jj,1)<circle_r) || (Cart_dist4(jj,1)<circle_r)

            
            Cart_dist(jj,1) = 0;
%             
%         elseif Cart_dist(jj,1) < radius(1) && Cart_dist(jj,1) > circle_r
%             
%             Cart_dist(jj,1) = 1;
            
        else
            
            Cart_dist(jj,1) = 1;
            
        end
        
    end
    
%     for jj = 1:length(Cart_dist(:,1))
%         
%         cir = circle_dist(Cart_data(jj,1),Cart_data(jj,2),circle_x,circle_y,r);
%         
%         if (cir < 0) && (Cart_dist(jj,1) > circle_r)
%             
%             Cart_dist(jj,1) = K;
%             
%         elseif (cir < 0) && (Cart_dist(jj,1) < circle_r)
%             
%             Cart_dist(jj,1) = -1;
%             
%         elseif (cir < 0) && (Cart_dist(jj,1) == circle_r)
%             
%             Cart_dist(jj,1) = 0;
%             
%         else
%             
%             Cart_dist(jj,1) = 1;
%             
%         end
%         
%     end
    
    % Adding labels to specific areas
%     
%     for j = 1:length(Cart_dist(:,1))
% 
%         if Cart_dist(j,1) < circle_r
%              Cart_dist(j,1) = -1;    
%         elseif Cart_dist(j,1) == circle_r
%              Cart_dist(j,1) = 0;
%         end
% 
%     end

%%%%%%%%%%%%%%% Adding Boundary Labels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     for k = 1:length(Cart_dist(:,1))
% 
%         if Cart_data(k,2) == 0   % boundary node
%            Cart_dist(k,1) = -2;
%         elseif Cart_data(k,2) == rectangle_y
%            Cart_dist(k,1) = -3;
%         elseif (Cart_data(k,1) == 0) || (Cart_data(k,1) == rectangle_x)
%            Cart_dist(k,1) = 0;
%         end
% 
%     end

%     Cart_dist = round(Cart_dist);
% 
%     figure(4*iii)
%     %%% [M_c, c] = contourf(Cart_data(:,1),Cart_data(:,2), level);
%     [M_c, c] = contourf(reshape(Cart_data(:,1), M), reshape(Cart_data(:,2), M), reshape(Cart_dist, M));
%     title('Stress Indicator Contour')
%     colorbar
%     shading interp;

    New_Cart = reshape(Cart_dist(:,1), M);

    % write sdf contour matrix
    
    sdf_cart_file = strcat(folder_path,'Composite_uniform_SDF_Cart_',idx,'.dat');
    fid2 = fopen(sdf_cart_file,'w');
    
    sdf_matrix = [Cart_data Cart_dist];
    sdf_matrix = sdf_matrix.';
    
    fprintf(fid2,'%d %d %d\n',sdf_matrix);
    
    fclose(fid2);
    
end