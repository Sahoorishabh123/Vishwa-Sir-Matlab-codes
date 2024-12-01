% Example intensity matrix I
I = [0.1, 0.3, 0.2; 
     0.4, 0.6, 0.5; 
     0.3, 0.2, 0.8];

% Example angle vector theta (in radians)
theta = linspace(0, 2*pi, size(I, 1));

% Find the maximum intensity value for each column (radius)
[max_values, max_indices] = max(I);

% Extract the angles corresponding to the maximum intensity values
angles_max_intensity = theta(max_indices);

% Display the result
disp(angles_max_intensity);
