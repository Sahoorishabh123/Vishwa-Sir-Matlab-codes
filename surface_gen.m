

[X,Y] = meshgrid(1:0.5:10,1:20);
Z = randi(3)*sin(X) + randi(3)*cos(Y);
surf(X,Y,Z);