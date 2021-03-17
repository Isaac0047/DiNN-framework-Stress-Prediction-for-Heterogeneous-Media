% This code aims at generating the Cartesian Map
% Point1 is the output Cartesian Map for analysis

function Point1 = Cartesian(rectangle_x, rectangle_y, n_x, n_y)


e_x = rectangle_x / n_x;
e_y = rectangle_y / n_y;

xvalue = e_x * [0:n_x];
yvalue = e_y * [0:n_y];

[X,Y] = meshgrid(xvalue, yvalue);
Point1 = [X(:) Y(:)];

% Tmp = zeros(n_y+1,n_x+1);

% surf(X,Y,Tmp)

