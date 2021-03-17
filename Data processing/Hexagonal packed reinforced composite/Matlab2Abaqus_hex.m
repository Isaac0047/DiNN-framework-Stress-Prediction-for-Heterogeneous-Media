function Matlab2Abaqus_center(Nodes,Node_Sets,Elements,Elements_Sets,Filename,NT_1,rectangle_x,rectangle_y)

fileID = fopen(Filename, 'w');

%Generate Nodes in Input File

fprintf(fileID,'*NODE, NSET=NODE\n');

[NNode, ND]=size(Nodes);

if ND==2  %2D                               
    
for i=1:1:NNode
    
fprintf(fileID,[num2str(i) ', ' num2str(Nodes(i,1)) ', ' num2str(Nodes(i,2)) '\n']); 

end

elseif ND==3  %3D
    
for i=1:1:NNode
    
fprintf(fileID,[num2str(i) ', ' num2str(Nodes(i,1)) ', ' num2str(Nodes(i,2)) ', ' num2str(Nodes(i,3)) '\n']); 

end    
    
end

mpc_x_1 = rectangle_x/2;
mpc_y_1 = rectangle_y + 0.001;

mpc_x_2 = rectangle_x + 0.001;
mpc_y_2 = rectangle_y/2;

fprintf(fileID,'*Node, NSET=MPC1\n');
fprintf(fileID,[num2str(9999) ', ' num2str(mpc_x_1) ', ' num2str(mpc_y_1) '\n']);

fprintf(fileID,'*Node, NSET=MPC2\n');
fprintf(fileID,[num2str(10000) ', ' num2str(mpc_x_2) ', ' num2str(mpc_y_2) '\n']);

% fprintf(fileID,'\n');

% Generating Node Set in the input file

for ii = 1:1:length(Node_Sets)
    
    fprintf(fileID,strcat('*NSET, NSET=',Node_Sets{ii}.Name,'\n'));
    
    for jj = 1:1:length(Node_Sets{ii}.Nodes) % Loop for the nodes in the node set
        
        IN = Node_Sets{ii}.Nodes(jj); % Nodes indices in node set
        
        NN = [num2str(IN) ', '];
        
        fprintf(fileID,[NN '\n']);
        
    end
    
%     fprintf(fileID,'\n');
       
end


%Generate Elements in Input File

for i=1:1:length(Elements_Sets)

    fprintf(fileID,strcat('*ELEMENT, ELSET=',Elements_Sets{i}.Name,', TYPE=',Elements_Sets{i}.Elements_Type,'\n'));

       for j=1:1:length(Elements_Sets{i}.Elements) %Loop for the elements in the elements set

           IE = Elements_Sets{i}.Elements(j); %Elements indices in elements sets
           IE1 = IE + (i-1)*NT_1;
           NNN = [num2str(IE1) ', '];

           for k=1:1:length(Elements{IE,i})   
    
               NNN=[NNN num2str(Elements{IE,i}(k)) ', '];

           end

           NNN=NNN(1:end-2);

           fprintf(fileID,[NNN '\n']);

       end

% fprintf(fileID,'\n');

end

fprintf(fileID,'**');

% Generating Element Set in the input file

% for ii = 1:1:length(Elements_Sub_Sets)
%     
%     fprintf(fileID,strcat('*ELEMENT, ELSET=',Elements_Sub_Sets{i}.Name,', TYPE=',Elements_Sub_Sets{i}.Elements_Type,'\n'));
%     
%     for jj = 1:1:length(Elements_Sub_Sets{ii}.Elements) % Loop for the nodes in the node set
%         
%         IEE = Elements_Sub_Sets{ii}.Elements(jj); % Nodes indices in node set
%         
%         EN = [num2str(IEE) ', '];
%         
%         fprintf(fileID,[EN '\n']);
%         
%     end
%     
% %     fprintf(fileID,'\n');
%        
% end
