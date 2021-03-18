clear all 
close all
fclose all
clc

%% User Defining basic geometry parameters

rectangle_x = 0.01;
rectangle_y = 0.01;

%% Starting loops to generate serious of geometry

data_set = 1000;

folder_path = 'C:\Temp_Abaqus\micro_meter_model\fix_hole_composite_small_random\';

for i_idx = 1:data_set   
    
i_idx = i_idx + 1000;
    
%% Circle center coordinates

% circle_r = sqrt(rectangle_x*rectangle_y*(0.4+0.2*rand(1,1))/pi);

circle_r = 0.001;
% circle_r = sqrt(rectangle_x*rectangle_y*(0.05+0.2*rand(1,1))/pi);

circle_x1 = 0.0020 + 0.0015*rand(1,1);
circle_y1 = 0.0020 + 0.0015*rand(1,1);

circle_x2 = 0.0065 + 0.0015*rand(1,1);
circle_y2 = 0.0020 + 0.0015*rand(1,1);

circle_x3 = 0.0020 + 0.0015*rand(1,1);
circle_y3 = 0.0065 + 0.0015*rand(1,1);

circle_x4 = 0.0065 + 0.0015*rand(1,1);
circle_y4 = 0.0065 + 0.0015*rand(1,1);

%% Create a decomposed solid model and include it in a PDE model

model = createpde;

% Define a circle in a rectangle, place these in one mtx

% Rectangle is code 3, 4 sides,
% followed by x-coordinates and then y-coordinates
R1 = [3,4,0,rectangle_x,rectangle_x,0,rectangle_y,rectangle_y,0,0]';

% Circle is code 1, center (.5,0), radius .2
C1 = [1,circle_x1,circle_y1,circle_r]';
C2 = [1,circle_x2,circle_y2,circle_r]';
C3 = [1,circle_x3,circle_y3,circle_r]';
C4 = [1,circle_x4,circle_y4,circle_r]';
% C5 = [1,circle_x5,circle_y5,circle_r]';

% Pad C1 with zeros to enable concatenation with R1
C1 = [C1;zeros(length(R1) - length(C1),1)];
C2 = [C2;zeros(length(R1) - length(C2),1)];
C3 = [C3;zeros(length(R1) - length(C3),1)];
C4 = [C4;zeros(length(R1) - length(C4),1)];
% C5 = [C5;zeros(length(R1) - length(C5),1)];

gm = [R1,C1,C2,C3,C4];

% Set formula
% '+' means union, '*' means intersection, '-' means difference
sf = 'R1-C1+C1-C2+C2-C3+C3-C4+C4';

% Names for the two geometric objects
ns = char('R1','C1','C2','C3','C4');
ns = ns';

% Create geometry
g = decsg(gm,sf,ns);

% Include the geometry in the model and plot it
geometryFromEdges(model,g);

% figure(2*i_idx-1)
% pdegplot(model,'EdgeLabels','on')
% axis equal

%% Generating mesh on the geometry

% generateMesh(model);
generateMesh(model,'GeometricOrder','linear','Hmax',0.0003);

% figure(2*i_idx)
% pdeplot(model)

%% Export the geometry into ABAQUS

id = num2str(i_idx);

pa1 = 'composite_model';
pa2 = strcat('_',id);
pa3 = '.inp';

Filename = strcat(folder_path,pa1,pa2,pa3);
% Filename='C:\Users\Haoti\Documents\MATLAB\Example.inp';

Nodes = model.Mesh.Nodes;
Nodes = Nodes';

% Clear extremely small values of nodal coordinate to zero

epsilon = 1e-8;

for i = 1:length(Nodes(:,1))
    for j = 1:length(Nodes(1,:))
        
        if abs(Nodes(i,j)) < epsilon
            
            Nodes(i,j) = 0;
            
        end
        
    end
end

mm = 1;
nn = 1;
ff = 1;
hh = 1;

clearvars Node_set_1
clearvars Node_set_2
clearvars Node_set_3
clearvars Node_set_4

for j = 1:length(Nodes(:,1))
    
    if Nodes(j,2) == rectangle_y
         Node_set_1(1,mm) = j;  
         mm = mm + 1;        
    elseif Nodes(j,2) == 0
         Node_set_2(1,nn) = j;
         nn = nn + 1;
    elseif Nodes(j,1) == 0
        Node_set_3(1,ff) = j;
        ff = ff + 1;
    elseif Nodes(j,1) == rectangle_x
        Node_set_4(1,hh) = j;
        hh = hh + 1;
    end
    
end

% n_nod_1 = length(Node_set_1(1,:));
% n_nod_2 = length(Node_set_2(1,:));

Node_Sets{1}.Name = 'Upper_Side';
Node_Sets{1}.Nodes = Node_set_1;

Node_Sets{2}.Name = 'Bottom_Side';
Node_Sets{2}.Nodes = Node_set_2;

Node_Sets{3}.Name  = 'Left_Side';
Node_Sets{3}.Nodes = Node_set_3;

Node_Sets{4}.Name  = 'Right_Side';
Node_Sets{4}.Nodes = Node_set_4;

%% Defining Element Sets

Element = model.Mesh.Elements;
L_x = length(Element(:,1));
L_y = length(Element(1,:));

clearvars Elements
clearvars pp
clearvars qq
clearvars Tmp1
clearvars Tmp2

pp = 1;
qq = 1;

circle1 = [circle_x1, circle_y1];
circle2 = [circle_x2, circle_y2];
circle3 = [circle_x3, circle_y3];
circle4 = [circle_x4, circle_y4];

for k=1:L_y
    
    node_id = Element(:,k);
    node_1  = [Nodes(node_id(1),1), Nodes(node_id(1),2)];
    node_2  = [Nodes(node_id(2),1), Nodes(node_id(2),2)];
    node_3  = [Nodes(node_id(3),1), Nodes(node_id(3),2)];
    
    centroid = 1/3 * (node_1+node_2+node_3);
        
    if (norm(centroid-circle1)<=circle_r) || (norm(centroid-circle2)<=circle_r) || (norm(centroid-circle3)<=circle_r) || (norm(centroid-circle4)<=circle_r)
            
        Tmp1{pp} = Element(:,k);
        pp = pp + 1;
            
    else
            
        Tmp2{qq} = Element(:,k);
        qq = qq + 1;
            
    end

end

n_ele_1 = pp-1;
n_ele_2 = qq-1;

for ii = 1:1:n_ele_1-1
    
    for ik = ii+1:1:n_ele_1-1
    
        if isequal(Tmp1{ik}, Tmp1{ii})
        
            Tmp1{ik} = [];
        
        end
    
    end
    
end

for jj = 1:1:n_ele_2-1
    
    for jk = jj+1:1:n_ele_2-1
    
        if isequal(Tmp2{jk},Tmp2{jj})
        
            Tmp2{jk} = [];
        
        end
    
    end
    
end

Tmp1 = Tmp1(~cellfun('isempty',Tmp1));
Tmp2 = Tmp2(~cellfun('isempty',Tmp2));

NT_1 = length(Tmp1);
NT_2 = length(Tmp2);

for mm = 1:1:NT_1
    
    Elements{mm,1} = Tmp1{mm};
    
end

for nn = 1:1:NT_2
    
    Elements{nn,2} = Tmp2{nn};

end

% n_ele_1 = size(Elements{1,:});
% n_ele_2 = size(Elements{2,:});

Elements_Sets{1}.Name = 'Set1';
Elements_Sets{1}.Elements_Type = 'CPS3';  
% Elements_Sets{1}.Elements_Type = 'CPS6';    
Elements_Sets{1}.Elements = 1:1:NT_1;

Elements_Sets{2}.Name = 'Set2';
Elements_Sets{2}.Elements_Type = 'CPS3';
% Elements_Sets{2}.Elements_Type = 'CPS6';   
Elements_Sets{2}.Elements = 1:1:NT_2;

Matlab2Abaqus_center(Nodes,Node_Sets,Elements,Elements_Sets,Filename,NT_1,rectangle_x,rectangle_y)

%% Write out Master file

% this program aims at writing Master.inp
f_w_name = strcat(folder_path,'Master_file_fix.inp');
fid = fopen(f_w_name);

cac = textscan(fid, '%s', 'Delimiter', '\n', 'CollectOutput', true);
% cac = fileread('Master_file_2.inp');

fclose(fid);

idx = num2str(i_idx);

path1 = folder_path;
path2 = 'New_Master_von';
path3 = strcat('_',idx,'.inp');

path = strcat(path1, '\', path2, path3);

fid2 = fopen(path,'w');
% fid2 = fopen('C:\Temp_Abaqus\New_Master.inp','w');

exname = strcat('*INCLUDE, INPUT=composite_model','_',idx,'.inp');

cac{1}{31} = exname;

for jj = 1 : length(cac{1})
    
    fprintf(fid2, '%s\n', cac{1}{jj});

end
% fprintf(fid2,'%s\n',cac{1})

fclose(fid2);

%% Write out circle center information

path_c = strcat(folder_path,'composite_cir_cen_',idx,'.dat');
fid3 = fopen(path_c,'w');

fprintf(fid3, '%d\n', circle_x1);
fprintf(fid3, '%d\n', circle_y1);
fprintf(fid3, '%d\n', circle_x2);
fprintf(fid3, '%d\n', circle_y2);
fprintf(fid3, '%d\n', circle_x3);
fprintf(fid3, '%d\n', circle_y3);
fprintf(fid3, '%d\n', circle_x4);
fprintf(fid3, '%d\n', circle_y4);
% fprintf(fid3, '%d\n', circle_x5);
% fprintf(fid3, '%d\n', circle_y5);
fprintf(fid3, '%d',   circle_r);

fclose(fid3);

end