
clear all
clc
close all

tic;

x = linspace(-7.005,7.005,2001);
[X,Y] = meshgrid(x,x);
[phi,r] = cart2pol(X,Y);


intensity_i_fill_1 = beam_func_i_fill(r,15,phi,2,2,0.1,0);
intensity_fill_1 = beam_func_fill(r,15,phi,2,2,0.1,0);

% s = load("save_random_surface2.mat");
% surf = s.randomSurf;
g = gaussian_surf(2001);

alpha = 0.1;   % resolution factor
betta = 1 ./ alpha;
% 
% 
 d1 = (360 .* betta)/(2*2);  %  dummy variable give location of peak in ideal superposition
% d2 = (360 .* betta)/(2*4);
% d3 = (360 .* betta)/(2*5); 
 d4 = (360 .* betta)/(2*6); 


 for R=1:1000
      for theta=d1/2:((360 * betta)+(d1/2))
          thetha = alpha .* theta;
          rows = (1001  -  round(R.*sind(thetha)));
          cols = (1001 +  round(R.*cosd(thetha)));

          C_fill_1(theta,R) = intensity_fill_1(rows,cols);
          C_ideal_fill_1(theta,R) = intensity_i_fill_1(rows,cols);

          %surf_polar(theta,R) = surf(rows,cols);
          surf_polar_g(theta,R) = g(rows,cols);

      end
 end

 % subplot(121),imagesc(C_ideal_fill_1);
 % subplot(122),imagesc(C_fill_1);

 %%

 
 % subplot(121),imagesc(C_ideal_fill_1);
 % subplot(122),imagesc(C_fill_1);
 regen_points = zeros(891,311);
%% 
%i = 0.70;
for i = 0.20:0.054:0.90
C_ideal_fill_1_og = C_ideal_fill_1;
lower_bound = i;
upper_bound = i+0.05;
C_ideal_fill_1(C_ideal_fill_1 >= lower_bound & C_ideal_fill_1 <= upper_bound) = 0;
%imagesc(C_ideal_fill_1);
desired_part_ideal = C_ideal_fill_1_og-C_ideal_fill_1;
%imagesc(desired_part_ideal);

% displaced case
C_fill_1_og = C_fill_1;
lower_bound = i;
upper_bound = i+0.05;
C_fill_1(C_fill_1 >= lower_bound & C_fill_1 <= upper_bound) = 0;
%imagesc(C_ideal_fill_1);
desired_part = C_fill_1_og-C_fill_1;
%subplot(122),imagesc(desired_part);
diff = desired_part - desired_part_ideal;

%imagesc(diff);



%% selecting a section of the petal

part_ideal = desired_part_ideal(350:1240,70:380);
part = desired_part(350:1240,70:380);
 imagesc(part_ideal);
 %imagesc(part);
diff_part = part - part_ideal;
 %imagesc(abs(diff_part));
%subplot(144), imagesc(diff_part);



%% trying different methods




% processing starts

A = diff_part;
% Initialize a vector to store counts for each columnA
num_columns = size(A, 2);
count = zeros(1, num_columns);  % Store count for each column


% Loop through each column
for col = 1:num_columns
    in_zero_sequence = false;  % Flag to track if we are in a zero sequence
    
    % Loop through each row in the current column
    for row = 1:size(A, 1)
        if A(row, col) == 0
            % If we encounter a zero and are not in a zero sequence, increment the count
            if ~in_zero_sequence
                count(col) = count(col) + 1;
                in_zero_sequence = true;  % We are now in a zero sequence
            end
        else
            % If we encounter a non-zero number, end the zero sequence
            in_zero_sequence = false;
        end
    end
end

%disp(count);


%% extracting the relevant regions

cnt = 0;
c = true;
str_cnt = zeros(1,size(count,2));
five_seq = true;
for row = 1:size(count,2)
    if count(row)==5
        if c
          d = row;
          c = false;
        end
        five_seq = true;
        if five_seq
            cnt = cnt+1;
         
        end
        
    
    else
         five_seq = false;
         str_cnt(row) = cnt;
         cnt = 0;
    end
   
   

end
disp(str_cnt);

[max_value, max_index] = max(str_cnt);
% disp(max_index);
% disp(max_value);

%%
B = zeros(size(diff_part));
% Define the specific column range

col_start = max_index-max_value+1;
col_end = max_index-1;

% Copy the specified columns to the new matrix
B(:, col_start:col_end) = diff_part(:, col_start:col_end);

imagesc(B);

%% extracting the clearest part
clr_part = abs(B);
% clr_part = abs(diff_part(100:770, max_index-max_value:max_index-1));
% imagesc(clr_part);



%% extracting the colm matrices

A = clr_part;

% Initialize a cell array to store non-zero series for each column
non_zero_series = cell(size(A, 2), 1);

% Loop through each column
for col = 1:size(A, 2)
    col_data = A(:, col);  % Extract the column
    consecutive_non_zero = [];  % To store one series of non-zero numbers
    series_list = {};  % To store all series for the current column
    
    % Loop through each element of the column
    for row = 1:length(col_data)
        if col_data(row) ~= 0
            % If it's a non-zero element, add it to the current sequence
            consecutive_non_zero = [consecutive_non_zero; (row)];
        else
            % If we encounter a zero and the current sequence is non-empty,
            % store the series and reset for the next series
            if ~isempty(consecutive_non_zero)
                series_list{end+1} = consecutive_non_zero;  % Store the series
                consecutive_non_zero = [];  % Reset for the next series
            end
        end
    end
    
    % If there is any remaining non-zero sequence at the end
    if ~isempty(consecutive_non_zero)
        series_list{end+1} = consecutive_non_zero;
    end
    
    % Store all series for the current column
    non_zero_series{col} = series_list;
    
    %Display the results for the current column
%     disp(['Column ', num2str(col), ':']);
%     for k = 1:length(series_list)
%         disp(['  Series ', num2str(k), ': ', num2str(series_list{k}')]);
%     end
end

%%




%% reconstructing part of the surface

gen_surf_test = zeros(size(clr_part));

for i = max_index-max_value+1:max_index-1
    x = non_zero_series{i};
    gen_surf_test(x{4}(size(x{4},1)),i) = x{4}(size(x{4},1))-x{3}(size(x{3},1));
    gen_surf_test(x{2}(size(x{2},1)),i) = x{2}(size(x{2},1))-x{1}(size(x{1},1));

end


regen_points = regen_points + gen_surf_test;

end

m = 0.28411285;
c = 0.0102449;


regen_points = ((regen_points-c)/m)*10^-9;
regen_points(regen_points<0)=0;
%regen_points = regen_points./max(max(regen_points));

%subplot(154),imagesc(regen_points);
%subplot(121),imagesc(regen_points);


%% showing the surface
 

%subplot(155),imagesc(surf(470:1340,100:380));


surf_polar_g_desired = surf_polar_g(350:1240,70:380);
%subplot(122),imagesc(surf_polar_g_desired);


%% interpolation
subplot(1,3,1),  mesh(regen_points);
non_zero_elements = regen_points(regen_points ~= 0);
[a,b] = find(regen_points);
[X, Y] = meshgrid(1:size(regen_points, 2), 1:size(regen_points, 1));
Z_spline = griddata(b, a, non_zero_elements, X, Y, 'cubic');
regen_points(regen_points == 0) = Z_spline(regen_points == 0);
regen_points(isnan(regen_points)) = 0;

%gen_surf = smoothdata(gen_surf);
subplot(1,3,2), mesh(surf_polar_g_desired);
subplot(1,3,3),  mesh(regen_points);











