clear all
clc
close all

tic;

x = linspace(-7.005,7.005,2001);
[X,Y] = meshgrid(x,x);
[phi,r] = cart2pol(X,Y);

%%
% 
% U = rand(2001,2001);
% U1 = (fft2(U));
% fil = r < 0.01;   % 0.005 mm
% U2 = fil.* U1 ;
% U3 = ifft2((U2));
% U3 = ifftshift(ifft2(U2));
% U4 = abs(U3);
% mesh(U4)
% U5 = U4 ./ max(U4);
% mesh(U5)
% U6 = 50 .* 10^(-9) .* U5;
% 
% mesh(U6);
% imagesc(U6);
% randomSurf = U6;
% imagesc(x,x,randomSurf)
% 
% % Assuming 'matrix' is your surface matrix
% matrix = randomSurf;
% histogram = histcounts(matrix(:), 'Normalization', 'probability');
% entropy_value = -sum(histogram .* log2(histogram + eps)); % eps to avoid log2(0)
% disp(entropy_value);
% 
% figure;
% mesh(x,x,randomSurf)
% random = load('save_random_surface.mat');
% 
% random2 = load('save_random_surface2.mat');
% surface=(random2.randomSurf); 
% 
% % Assuming 'matrix' is your surface matrix
% matrix = surface;
% histogram = histcounts(matrix(:), 'Normalization', 'probability');
% entropy_value = -sum(histogram .* log2(histogram + eps)); % eps to avoid log2(0)
% disp(entropy_value);

%% gaussian surface

%surface = gaussian_surf(2001);
%imagesc(surface);
%%

%rs = load("randomSurf", "U6");
%rs = rs.U6;
%delta = rs

lambda = 633e-9; 
k = 2.*pi./lambda; 

% Model Parameters
 l1 = 2;    % topological charge;
 n1 = 2;  % radial index; n=|l|,|l|+2,|l|+4 ...
 D = sqrt(2);   % is a constant for normalization;

 %dis = 1*10^-9;
 % dis = 0;

l2 = 4;
n2 = 14;
Z = 0.0007;  %[-] a XY-slice in the z-direction
Z2 = 40;
% Analytical functions
% gauss = @(r) exp(-r.^2./0.006^2);
% G = @(r,z) D./sqrt(0.3*(1+z.^2)).*exp(-r.^2./(0.3^2*(1+z.^2))).*exp(-1i/4*(z.*r.^2)./(0.3^2*(1+z.^2)));
% A = @(r,Z) (sqrt(2)*r./(0.3*sqrt(1+Z.^2))).^abs(l1).*LaguerreL((n1-abs(l1))/2,abs(l1),2*r.^2./(0.3^2*(1+Z.^2)));
% PHI = @(th) exp(1i*l1*th);
% PHIM = @(th) exp(1i*-l1*th);
% PSI = @(z) exp(-1i*(n1+1)*atan(z));
% pol = @(th) exp(-1i*th);
% P =  G(r,Z).*A(r,Z).*PHI(phi).*PSI(Z).*exp(-1i*0);%.*exp(1i*2*k.*gauss);
% PMg = G(r,Z).*A(r,Z).*PHIM(phi).*PSI(Z).*exp(-1i*0).*exp(1i*2*k*surface);
% % PM = G(r,Z).*A(r,Z).*PHIM(phi).*PSI(Z).*exp(-1i*0).*exp(1i*2*k*randomSurf);
% PM = G(r,Z).*A(r,Z).*PHIM(phi).*PSI(Z).*exp(-1i*0);
% Sg = P+PMg; %.*exp(1i*2*k*3);
% S  = P+PM;
% Ag=abs(Sg).^2;
% A= abs(S).^2;
% norm_A = A./max(max(A));
% norm_Ag = Ag./max(max(Ag));
% 
% G2 = @(r,z) D./sqrt(1.4*(1+z.^2)).*exp(-r.^2./((1.4)^2*(1+z.^2))).*exp(-1i/4*(z.*r.^2)./((1.4)^2*(1+z.^2)));
% A2 = @(r,Z) (sqrt(2)*r./(1.4*sqrt(1+Z.^2))).^abs(l2).*LaguerreL((n2-abs(l2))/2,abs(l2),2*r.^2./((1.4)^2*(1+Z.^2)));
% PHI2 = @(th) exp(1i*l2*th);
% PHIM2 = @(th) exp(1i*-l2*th);
% PSI2 = @(z) exp(-1i*(n2+1)*atan(z));
% P2 =  G2(r,Z2).*A2(r,Z2).*PHI2(phi).*PSI2(Z2).*exp(-1i*0);
% PMg2 = G2(r,Z2).*A2(r,Z2).*PHIM2(phi).*PSI2(Z2).*exp(-1i*0).*exp(1i*2*k*surface);
% PM2 = G2(r,Z2).*A2(r,Z2).*PHIM2(phi).*PSI2(Z2).*exp(-1i*0);
% Sg2 = P2+PMg2; %.*exp(1i*2*k*3);
% S2  = P2+PM2;
% Ag2=abs(Sg2).^2;
% A2= abs(S2).^2;
% norm_A2 = A2./max(max(A2));
% norm_Ag2 = Ag2./max(max(Ag2));

%intensity_i = beam_func_i(r,Z2,phi,l1,n1,0.1) + beam_func_i(r,Z2,phi,l1,n1,0.5)+beam_func_i(r,Z2,phi,l1,n1,1.0);
%intensity = beam_func(r,Z2,phi,l1,n1,0.1) + beam_func(r,Z2,phi,l1,n1,0.5)+beam_func(r,Z2,phi,l1,n1,1.0);
  
 % intensity_i = beam_func_i(r,Z2,phi,l1,n1,0.1);
 % intensity = beam_func(r,Z2,phi,l1,n1,0.1);
 % 
 intensity_i_fill_1 = beam_func_i_fill(r,15,phi,2,2,0.1,0);
 intensity_fill_1 = beam_func_fill(r,15,phi,2,2,0.1,0);

 imagesc(intensity_fill_1);
 % 
 % intensity_i_fill_2 = beam_func_i_fill(r,25,phi,4,4,0.1,0);
 % intensity_fill_2 = beam_func_fill(r,25,phi,4,4,0.1,0);
 % 
 % intensity_i_fill_3 = beam_func_i_fill(r,35,phi,5,5,0.1,0);
 % intensity_fill_3 = beam_func_fill(r,35,phi,5,5,0.1,0);
 % 
 % intensity_i_fill_4 = beam_func_i_fill(r,35,phi,6,6,0.1,0);
 % intensity_fill_4 = beam_func_fill(r,35,phi,6,6,0.1,0);


 % diff = (norm_A2 - norm_Ag2);
% res = imfuse(norm_Ag,norm_A,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
% imshow(res);
% P = impixel(res);


%%


%Plot a single slice of the presure profile
 figure(1); fontsize=12;
 set(gcf,'position',[100,100,600,200])

  %subplot(121), imagesc(x,x,flipud(intensity_i_fill_1)); colorbar; axis tight

 xlabel('x','interpreter','latex','fontsize',fontsize);
 ylabel('y','interpreter','latex','fontsize',fontsize);
 title('magnitude','interpreter','latex','fontsize',fontsize);
 %subplot(122),imagesc(x,x,flipud(intensity_fill_1)); colorbar; axis tight
 xlabel('x','interpreter','latex','fontsize',fontsize);
 ylabel('y','interpreter','latex','fontsize',fontsize);
 title('phase at $t_0$','interpreter','latex','fontsize',fontsize);
 %subplot(133), imagesc(x,x,flipud(diff)); colorbar; axis tight


 % I_rot = imread('test_img_rotated.png');
 % I_ideal = imread('test_img_ideal.png');

 %  subplot(121), imhist(I_ideal);
 %  subplot(122), imhist(I_rot);

 
 


%% polar coordinate
 alpha = 0.1;   % resolution factor
betta = 1 ./ alpha;
% 
% 
 d1 = (360 .* betta)/(2*3);  %  dummy variable give location of peak in ideal superposition
% d2 = (360 .* betta)/(2*4);
% d3 = (360 .* betta)/(2*5); 
 d4 = (360 .* betta)/(2*6); 
% 
% 
% 
for R=1:1000
      for theta=d1/2:((360 * betta)+(d1/2))
          thetha = alpha .* theta;
          rows = (1001  -  round(R.*sind(thetha)));
          cols = (1001 +  round(R.*cosd(thetha)));

          C_fill_1(theta,R) = intensity_fill_1(rows,cols);
          C_ideal_fill_1(theta,R) = intensity_i_fill_1(rows,cols);

          %surf_polar(theta,R) = surface(rows,cols);

      end
end
% %%
% for R=1:1000
%       for theta=d2/2:((360 * betta)+(d2/2))
%           thetha = alpha .* theta;
%           rows = (1001  -  round(R.*sind(thetha)));
%           cols = (1001 +  round(R.*cosd(thetha)));
% 
% 
%           C_fill_2(theta,R) = intensity_fill_2(rows,cols);
%           C_ideal_fill_2(theta,R) = intensity_i_fill_2(rows,cols);
% 
%           surf_polar(theta,R) = surface(rows,cols);
% 
%       end
% end
% %%
% for R=1:1000
%       for theta=d3/2:((360 * betta)+(d3/2))
%           thetha = alpha .* theta;
%           rows = (1001  -  round(R.*sind(thetha)));
%           cols = (1001 +  round(R.*cosd(thetha)));
% 
%           C_fill_3(theta,R) = intensity_fill_3(rows,cols);
%           C_ideal_fill_3(theta,R) = intensity_i_fill_3(rows,cols);
% 
%           surf_polar(theta,R) = surface(rows,cols);
% 
%       end
% end
% %%
% for R=1:1000
%       for theta=d4/2:((360 * betta)+(d4/2))
%           thetha = alpha .* theta;
%           rows = (1001  -  round(R.*sind(thetha)));
%           cols = (1001 +  round(R.*cosd(thetha)));
% 
%           %C_fill_4(theta,R) = intensity_fill_4(rows,cols);
%           %C_ideal_fill_4(theta,R) = intensity_i_fill_4(rows,cols);
%           surf_polar(theta,R) = surface(rows,cols);
% 
%       end
% end
% 
% surf_polar(:,906:1000)=0;
% ideal case
C_ideal_fill_1_og = C_ideal_fill_1;
lower_bound = 0.85;
upper_bound = 0.9;
C_ideal_fill_1(C_ideal_fill_1 >= lower_bound & C_ideal_fill_1 <= upper_bound) = 0;
%imagesc(C_ideal_fill_1);
desired_part_ideal = C_ideal_fill_1_og-C_ideal_fill_1;
%subplot(121),imagesc(desired_part_ideal);

% displaced case
C_fill_1_og = C_fill_1;
lower_bound = 0.85;
upper_bound = 0.9;
C_fill_1(C_fill_1 >= lower_bound & C_fill_1 <= upper_bound) = 0;
%imagesc(C_ideal_fill_1);
desired_part = C_fill_1_og-C_fill_1;
%subplot(122),imagesc(desired_part);
diff = desired_part - desired_part_ideal;

subplot(121),imagesc(diff);

%% selecting particular range

A = diff*10000;  
lower_bound = -1;

upper_bound = 1; 


% Create a mask for elements within the specified range
mask = (A >= lower_bound ) & (A <= upper_bound) ;

% Set elements outside the range to zero while keeping the original matrix dimensions
A_filtered = A;  % Start with a copy of A
A_filtered(~mask) = 0;  % Set elements outside the range to zero





 subplot(122),imagesc(A_filtered);

%% selecting a section of the petal

part_ideal = desired_part_ideal(600:1100,150:270);
part = desired_part(600:1100,150:270);
subplot(141), imagesc(part_ideal);
subplot(142), imagesc(part);
diff_part = part - part_ideal;
subplot(143), imagesc(abs(diff_part));
subplot(144), imagesc(diff_part);



%% trying different methods

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

disp(count);

%% extracting the relevant regions

cnt = 0;
str_cnt = zeros(1,size(count,2));
five_seq = true;
for row = 1:size(count,2)
    if count(row)==5
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

%% extracting the clearest part
clr_part = abs(diff_part(1:501,max_index-max_value:max_index-1));
imagesc(clr_part);

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
    
    % Display the results for the current column
    % disp(['Column ', num2str(col), ':']);
    % for k = 1:length(series_list)
    %     disp(['  Series ', num2str(k), ': ', num2str(series_list{k}')]);
    % end
end

%% checking the shifts

x = non_zero_series{20};
disp(x{3}(size(x{3})));
disp(x{4}(size(x{4},1)));

disp(x{4}(size(x{4}))-x{3}(size(x{3})));

disp(x{1}(size(x{1})));
disp(x{2}(size(x{2})));

disp(x{2}(size(x{2}))-x{1}(size(x{1})));



%% reconstructing part of the surface

gen_surf_test = zeros(size(clr_part));

for i = 1:size(clr_part,2)
    x = non_zero_series{i};
    gen_surf_test(x{4}(size(x{4},1)),i) = x{4}(size(x{4},1))-x{3}(size(x{3},1));
    gen_surf_test(x{2}(size(x{2},1)),i) = x{2}(size(x{2},1))-x{1}(size(x{1},1));

end

%imagesc(gen_surf_test);




%%

% threshold = 0.80; % 0.15 for the whole beam
% 
%      % C_ideal(C_ideal<threshold)  = 0.00;
%      % C(C<threshold)  = 0.00;
% 
%      C_ideal_fill_1(C_ideal_fill_1<threshold)  = 0.00;
%      C_fill_1(C_fill_1<threshold)  = 0.00;
% 
%      % C_ideal_fill_2(C_ideal_fill_2<threshold)  = 0.00;
%      % C_fill_2(C_fill_2<threshold)  = 0.00;
% 
%      C_ideal_fill_3(C_ideal_fill_3<threshold)  = 0.00;
%      C_fill_3(C_fill_3<threshold)  = 0.00;
% 
%      C_ideal_fill_4(C_ideal_fill_4<threshold)  = 0.00;
%      C_fill_4(C_fill_4<threshold)  = 0.00;


     % C_ideal = C_ideal + C_ideal_fill;
     % C = C + C_fill;

      % subplot(1,2,1);
      % imagesc(beam_show(r,15,phi,6,6,0));
      % subplot(1,2,2);
      % imagesc(surf_polar(500:3700,200:600));

      %%
      
%      %Find the columns (x-axis indices) with non-zero values
%      nonZeroColumns = any(C_ideal_fill_1 ~= 0, 1);
% 
%      % Get the range of x-axis indices with non-zero values
%      nonZeroIndices = find(nonZeroColumns);
%      r_min_1 = min(nonZeroIndices);
%      r_max_1 = max(nonZeroIndices);
% %%
%       %Find the columns (x-axis indices) with non-zero values
%      nonZeroColumns = any(C_ideal_fill_2 ~= 0, 1);
% 
%      % Get the range of x-axis indices with non-zero values
%      nonZeroIndices = find(nonZeroColumns);
%      r_min_2 = min(nonZeroIndices);
%      r_max_2 = max(nonZeroIndices);
% %%
%       %Find the columns (x-axis indices) with non-zero values
%      nonZeroColumns = any(C_ideal_fill_3 ~= 0, 1);
% 
%      % Get the range of x-axis indices with non-zero values
%      nonZeroIndices = find(nonZeroColumns);
%      r_min_3 = min(nonZeroIndices);
%      r_max_3 = max(nonZeroIndices);
% %%
%       %Find the columns (x-axis indices) with non-zero values
%      nonZeroColumns = any(C_ideal_fill_4 ~= 0, 1);
% 
%      % Get the range of x-axis indices with non-zero values
%      nonZeroIndices = find(nonZeroColumns);
%      r_min_4 = min(nonZeroIndices);
%      r_max_4 = max(nonZeroIndices);
     

     %% filling gaps(l=2)

     
     %gen_surf = reg_disp(0,20);

     %disp(gen_surf(500:3500,r_min:r_max-10));
   
      
    %disp(gen_surf);
        
    
      
%%


% desired_part_ideal = C_ideal((360.*betta/(4*l1)):(3*360.*betta/(4*l1)),1:1000);
% subplot(1,3,1);
% imagesc(desired_part_ideal);
% desired_part = C((360.*betta/(4*l1)):(3*360.*betta/(4*l1)),1:1000);
% subplot(1,3,2);
% 
% imagesc(desired_part);
% diff = (desired_part-desired_part_ideal);
% subplot(1,3,3);
% imagesc(diff);
% % % 
% maximum_ideal = max(max(desired_part_ideal));
% [x_ideal,y_ideal]=find(desired_part_ideal == maximum_ideal);
% data_i = [x_ideal,y_ideal];
% first_col_i = data_i(:,1);
% median_data_ideal = median(first_col_i);
% %disp(median_data_ideal);
% %disp([x_ideal./((360 .* betta)/(2*l2)),y_ideal]);
% %disp([x_ideal,y_ideal]);
% maximum_disp = max(max(desired_part));
% [x_disp,y_disp]=find(desired_part == maximum_disp);
% data = [x_disp,y_disp];
% first_col = data(:,1);
% median_data_disp = median(first_col);
% %disp(median_data_disp);
% %disp([x_disp./((360 .* betta)/(2*l2)),y_disp]);
% %disp([x_disp,y_disp]);
% 
% %disp(abs(median_data_disp-median_data_ideal));
% shifts(d/2)= abs(median_data_disp-median_data_ideal);




%  %% maxima of the diff
% 
% maximum_diff = max(max(-(desired_part-desired_part_ideal)));
% [x_disp,y_disp]=find(-(desired_part-desired_part_ideal) == maximum_diff);
% data_diff = [x_disp,y_disp];
% % disp(data_diff);
% first_col_diff = data_diff(:,1);
% median_data_diff = median(first_col_diff);
% 
% disp(median_data_diff);
%%
% edges_I0 = edge(C_ideal(18:54,200:300), 'canny');
% edges_I_displaced = edge(C(18:54,200:300), 'canny');
% 
% figure;
% subplot(1, 2, 1);
% imshow(edges_I0);
% title('Edges of Ideal Case');
% 
% subplot(1, 2, 2);
% imshow(edges_I_displaced);
% title('Edges of Displaced Case');

%%
% data = [x,y];
% first_col = data(:,1);
% median_data = median(first_col);
% disp(median_data);

%% loop for the whole matrices

gen_surf_1 = zeros(3750,1000);
gen_surf_2 = zeros(3750,1000);
gen_surf_3 = zeros(3750,1000);
gen_surf_4 = zeros(3750,1000);
gen_surf_5 = zeros(3750,1000);
gen_surf_6 = zeros(3750,1000);
gen_surf_7 = zeros(3750,1000);
gen_surf_8 = zeros(3750,1000);
gen_surf_9 = zeros(3750,1000);
gen_surf_10 = zeros(3750,1000);
gen_surf_11 = zeros(3750,1000);
gen_surf_12 = zeros(3750,1000);
gen_surf_13 = zeros(3750,1000);
gen_surf_14 = zeros(3750,1000);
gen_surf =  zeros(3750,1000);


% s_1 = round((r_max_1-r_min_1)/3);
% s_2 = round((r_max_2-r_min_2)/3);
% s_3 = round((r_max_3-r_min_3)/4);
% s_4 = round((r_max_4-r_min_4)/3);



% %%
% for rad = r_min_4:s_4:r_max_4
%    if(rad+s_4)>1000
%          break;
%    end 
%    %disp(rad);
% 
%     for th = d4/2:d4:((360 * betta)+(d4/2)) 
% 
%        if(th+(d4))>((360 * betta)+(d4/2))
%             break;
%        end 
% 
%        desired_part_ideal = C_ideal_fill_4(th:th+(d4),rad:rad+s_4);
%        desired_part = C_fill_4(th:th+(d4),rad:rad+s_4);
%        % diff = (desired_part-desired_part_ideal);
%        maximum = max(max(desired_part));
%        [x,y]=find(desired_part == maximum);
%        data = [x,y];
%        first_col = data(:,1);
%        median_data = median(first_col);
% 
%        maximum_i = max(max(desired_part_ideal));
%        [x_i,y_i]=find(desired_part_ideal == maximum_i);
% 
%        data_i = [x_i,y_i];
%        first_col_i = data_i(:,1);
%        median_data_i = median(first_col_i);
%        %disp(median_data_i);
%        shift = abs(median_data - median_data_i)/10;
%         %disp(shift);
%        % disp(median_data);
%        if shift ~= 0
%           gen_surf_4(th+(d4/2)-50,rad) = ((shift-4.08163244e-05)/9.46854742e-02)*10^-9;
% 
%        else
%           gen_surf_4(th+(d4/2)-50,rad) = 0;
%        end
% 
% 
% 
%        %disp(th);
% 
%     end
% end
% 
% %%
% 
% for rad = r_min_3:s_3:r_max_3
% 
%    if(rad+s_3)>1000
%          break;
%    end 
%    %disp(rad);
% 
%     for th = d3/2:d3:((360 * betta)+(d3/2))  
% 
%        if(th+(d3))>((360 * betta)+(d3/2))
%             break;
%        end 
% 
%        desired_part_ideal = C_ideal_fill_3(th:th+(d3),rad:rad+s_3);
%        desired_part = C_fill_3(th:th+(d3),rad:rad+s_3);
%        % diff = (desired_part-desired_part_ideal);
%        maximum = max(max(desired_part));
%        [x,y]=find(desired_part == maximum);
%        data = [x,y];
%        first_col = data(:,1);
%        median_data = median(first_col);
% 
%        maximum_i = max(max(desired_part_ideal));
%        [x_i,y_i]=find(desired_part_ideal == maximum_i);
% 
%        data_i = [x_i,y_i];
%        first_col_i = data_i(:,1);
%        median_data_i = median(first_col_i);
%        %disp(median_data_i);
%        shift = abs(median_data - median_data_i)/10;
%         %disp(shift);
%        % disp(median_data);
%        if shift ~= 0
%           gen_surf_3(th+(d3/2)-50,rad) = ((shift-0.01322449)/0.11356423)*10^-9;
% 
%        else
%           gen_surf_3(th+(d3/2)-50,rad) = 0;
%        end
% 
% 
% 
%        %disp(th);
% 
%     end
% end
% 
% 
% %%
% 
% 
% 
% for rad = r_min_1:s_1:r_max_1
% 
%    if(rad+s_1)>1000
%          break;
%    end 
%    %disp(rad);
% 
%     for th = d1/2:d1:((360 * betta)+(d1/2)) 
% 
%        if(th+(d1))>((360 * betta)+(d1/2))
%             break;
%        end 
% 
%        desired_part_ideal = C_ideal_fill_1(th:th+(d1),rad:rad+s_1);
%        desired_part = C_fill_1(th:th+(d1),rad:rad+s_1);
%        % diff = (desired_part-desired_part_ideal);
%        maximum = max(max(desired_part));
%        [x,y]=find(desired_part == maximum);
%        data = [x,y];
%        first_col = data(:,1);
%        median_data = median(first_col);
% 
%        maximum_i = max(max(desired_part_ideal));
%        [x_i,y_i]=find(desired_part_ideal == maximum_i);
% 
%        data_i = [x_i,y_i];
%        first_col_i = data_i(:,1);
%        median_data_i = median(first_col_i);
%        %disp(median_data_i);
%        shift = abs(median_data - median_data_i)/10;
%         %disp(shift);
%        % disp(median_data);
%        if shift ~= 0
%           gen_surf_1(th+(d1/2)-50,rad) = ((shift-0.0102449)/0.28411285)*10^-9;
% 
%        else
%           gen_surf_1(th+(d1/2)-50,rad) = 0;
%        end
% 
% 
% 
%        %disp(th);
% 
%     end
% end
% 
% %%
% 
% for rad = r_min_2:s_2:r_max_2
% 
%    if(rad+s_2)>1000
%          break;
%    end 
%    %disp(rad);
% 
%     for th = d2/2:d2:((360 * betta)+(d2/2)) 
% 
%        if(th+(d2))>((360 * betta)+(d2/2))
%             break;
%        end 
% 
%        desired_part_ideal = C_ideal_fill_2(th:th+(d2),rad:rad+s_2);
%        desired_part = C_fill_2(th:th+(d2),rad:rad+s_2);
% 
%        % diff = (desired_part-desired_part_ideal);
%        maximum = max(max(desired_part));
%        [x,y]=find(desired_part == maximum);
%        data = [x,y];
%        first_col = data(:,1);
%        median_data = median(first_col);
% 
%        maximum_i = max(max(desired_part_ideal));
%        [x_i,y_i]=find(desired_part_ideal == maximum_i);
%        data_i = [x_i,y_i];
%        first_col_i = data_i(:,1);
%        median_data_i = median(first_col_i);
%        shift = abs(median_data - median_data_i)/10;
%         %disp(shift);
%         %disp(median_data_i);
%        if shift ~= 0
%           gen_surf_2(th+(d2/2)-50,rad) = ((shift-0.01355102)/0.14193037)*10^-9;
% 
%        else
%           gen_surf_2(th+(d2/2)-50,rad) = 0;
%        end
% 
%        %disp(th);
% 
%     end
% end

%%  beam_checking


  % test=(beam_show(r,15,phi,3,3,0));
  % imagesc(test);

%imagesc(colorbar);


%%
    
for i = 6:2:30
    gen_surf_1 = gen_surf_1+reg_disp(0,2,2,i,3);
    gen_surf_2 = gen_surf_2+reg_disp(0,3,3,i,3);
    gen_surf_3 = gen_surf_3+reg_disp(0,4,4,i,3);
    gen_surf_4 = gen_surf_4+reg_disp(0,5,5,i,3);
    gen_surf_5 = gen_surf_5+reg_disp(0,6,6,i,3);
    gen_surf_6 = gen_surf_6+reg_disp(0,9,9,i,3);


end
%%
% gen_surf_1 = reg_disp(0,2,2,25,3)+reg_disp(1/20,2,2,25,3)+reg_disp(1/15,2,2,25,3)+reg_disp(1/10,2,2,25,3);
% gen_surf_2 = reg_disp(0,4,4,14,4);
% gen_surf_3 = reg_disp(0,9,9,25,2)+reg_disp(1/20,9,9,25,2)+reg_disp(1/10,9,9,25,2);
% gen_surf_4 = reg_disp(0,9,9,6,10);
% gen_surf_5 = reg_disp(0,9,9,31,4)+reg_disp(1/15,9,9,31,5)+reg_disp(1/10,9,9,31,5)+reg_disp(1/20,9,9,31,5);
% gen_surf_6 = reg_disp(0,5,5,15,5);
% gen_surf_7 = reg_disp(0,6,6,15,5)+reg_disp(0,6,6,17,5)+reg_disp(0,6,6,20,5)+reg_disp(0,6,6,22,5);
% gen_surf_8 = reg_disp(0,9,9,30,10)+reg_disp(0,9,9,15,5)+reg_disp(0,9,9,17,5)+reg_disp(0,9,9,19,5)+reg_disp(0,9,9,21,5);
% gen_surf_9 = reg_disp(0,2,2,6,10)+reg_disp(1/20,2,2,6,3)+reg_disp(1/15,2,2,6,3);
%gen_surf_10 = reg_disp(0,2,2,4,10)+reg_disp(0,3,3,4,10);
% gen_surf_11 = reg_disp(0,2,2,39,12)+reg_disp(0,2,2,16,12)+reg_disp(0,2,2,18,12);
% gen_surf_12 = reg_disp(0,2,2,37,12)+reg_disp(0,2,2,14,12)+reg_disp(0,2,2,12,12)+reg_disp(0,2,2,10,12);
% 
% gen_surf_14 = reg_disp(0,3,3,33,2)+reg_disp(0,2,2,38,2)+reg_disp(0,9,9,29,10)+reg_disp(0,9,9,27,10)+reg_disp(0,9,9,25,10);

%gen_surf = gen_surf_1+gen_surf_2+gen_surf_3+gen_surf_4;
%%

 % Define matrices
matrix1 = gen_surf_1;
matrix2 = gen_surf_2;
matrix3 = gen_surf_3;
matrix4 = gen_surf_4;
matrix5 = gen_surf_5;
matrix6 = gen_surf_6;
%matrix10 = gen_surf_10;

%matrix13 = gen_surf_13;
% matrix14 = gen_surf_14;





% List of matrices
list_of_matrices = {matrix1,matrix2,matrix3,matrix4,matrix5,matrix6};

% Number of matrices
number_matrices = 6;

% Get dimensions
[rows, columns] = size(matrix1);

% Initialize coefficient matrix and sum matrix
coefficient_matrix = zeros(rows, columns);
sum_matrix = zeros(rows, columns);

% Calculate coefficient matrix
for k = 1:number_matrices
    matrix = list_of_matrices{k};
    for i = 1:rows
        for j = 1:columns
            if matrix(i, j) > 0
                coefficient_matrix(i, j) = coefficient_matrix(i, j) + 1;
            end
        end
    end
end

% Calculate sum matrix
for k = 1:number_matrices
    matrix = list_of_matrices{k};
    for i = 1:rows
        for j = 1:columns
            if matrix(i, j) > 0
                sum_matrix(i, j) = sum_matrix(i, j) + matrix(i, j);
            end
        end
    end
end

% Calculate average
for i = 1:rows
    for j = 1:columns
        if coefficient_matrix(i, j) > 0
            sum_matrix(i, j) = sum_matrix(i, j) / coefficient_matrix(i, j);
        end
    end
end

% Display result
gen_surf=sum_matrix;

%%

% surf_2 = load("gen_surf(l=2).mat");
% surf_3 = load("gen_surf(l=3).mat");
% surf_4 = load("gen_surf(l=4).mat");
% surf_5 = load("gen_surf(l=5).mat");
% 
% 
% imagesc(surf_3.gen_surf);


 %gen_surf = gen_surf_1+gen_surf_2+gen_surf_3+gen_surf_4;


% disp(max(max(gen_surf_1)));
% disp(max(max(gen_surf_2)));

     % gen_surf_4(gen_surf_4<0) = 0.00;
     % [a,b] = find(gen_surf_4);

    % disp([a,b]);
    
     %non_zero_elements = gen_surf_4(gen_surf_4 ~= 0);
     %disp(non_zero_elements);
% disp(gen_surf);


%% interpolation
% 
% [X, Y] = meshgrid(1:size(gen_surf_4, 2), 1:size(gen_surf_4, 1));
% Z_spline = griddata(b, a, non_zero_elements, X, Y, 'cubic');
% gen_surf_4(gen_surf_4 == 0) = Z_spline(gen_surf_4 == 0);

% %gen_surf = smoothdata(gen_surf);
%subplot(1,2,1), imagesc(surf_polar(500:3700,1:1000));
%subplot(1,2,2),  imagesc(gen_surf_4(500:3700,1:1000));
%des = surf_polar(1000:1900,r_min:r_max-10);

%% total interpolation after individually interpolating each part
%
% gen_surf_4(gen_surf_4<0) = 0.00;
% [a,b] = find(gen_surf_4);
% 
% non_zero_elements = gen_surf_4(gen_surf_4 ~= 0);
% 
% 
% [X, Y] = meshgrid(1:size(gen_surf_4, 2), 1:size(gen_surf_4, 1));
% Z_spline = griddata(b, a, non_zero_elements, X, Y, 'cubic');
% gen_surf_4(gen_surf_4 == 0) = Z_spline(gen_surf_4 == 0);
% 
% gen_surf_4(isnan(gen_surf_4)) = 0;
% 
% %
% 
% gen_surf_3(gen_surf_3<0) = 0.00;
% [a,b] = find(gen_surf_3);
% 
% non_zero_elements = gen_surf_3(gen_surf_3 ~= 0);
% 
% 
% [X, Y] = meshgrid(1:size(gen_surf_3, 2), 1:size(gen_surf_3, 1));
% Z_spline = griddata(b, a, non_zero_elements, X, Y, 'cubic');
% gen_surf_3(gen_surf_3 == 0) = Z_spline(gen_surf_3 == 0);
% 
% gen_surf_3(isnan(gen_surf_3)) = 0;
% 
% %
% 
% gen_surf_2(gen_surf_2<0) = 0.00;
% [a,b] = find(gen_surf_2);
% 
% non_zero_elements = gen_surf_2(gen_surf_2 ~= 0);
% 
% 
% [X, Y] = meshgrid(1:size(gen_surf_2, 2), 1:size(gen_surf_2, 1));
% Z_spline = griddata(b, a, non_zero_elements, X, Y, 'cubic');
% gen_surf_2(gen_surf_2 == 0) = Z_spline(gen_surf_2 == 0);
% 
% gen_surf_2(isnan(gen_surf_2)) = 0;
% 
% %
% 
% gen_surf_1(gen_surf_1<0) = 0.00;
% [a,b] = find(gen_surf_1);
% 
% non_zero_elements = gen_surf_1(gen_surf_1 ~= 0);
% 
% 
% [X, Y] = meshgrid(1:size(gen_surf_1, 2), 1:size(gen_surf_1, 1));
% Z_spline = griddata(b, a, non_zero_elements, X, Y, 'cubic');
% gen_surf_1(gen_surf_1 == 0) = Z_spline(gen_surf_1 == 0);
% 
% gen_surf_1(isnan(gen_surf_1)) = 0;

%

%gen_surf = gen_surf_1+gen_surf_2+gen_surf_3;

%imagesc(gen_surf);
%%
gen_surf(gen_surf<0) = 0.00;
[a,b] = find(gen_surf);

non_zero_elements = gen_surf(gen_surf ~= 0);


%%



[X, Y] = meshgrid(1:size(gen_surf, 2), 1:size(gen_surf, 1));

 Z_spline = griddata(b, a, non_zero_elements, X, Y, 'cubic');
 Z_spline = fillmissing(Z_spline, 'nearest');

 gen_surf(gen_surf == 0) = Z_spline(gen_surf == 0);
% 
 gen_surf(isnan(gen_surf))=0;

 %%

 %Find the columns (x-axis indices) with non-zero values
     nonZeroColumns = any(gen_surf ~= 0, 1);

     % Get the range of x-axis indices with non-zero values
     nonZeroIndices = find(nonZeroColumns);
     r_min = min(nonZeroIndices);
     r_max = max(nonZeroIndices);

     
   % subplot(1,4,1), imagesc(surf_polar(500:3700,1:1000));
   % subplot(1,4,2),  imagesc(gen_surf(500:3700,1:1000));




%% converting surface from polar to cartesian
 
for R0=1:1000
    for j=d4/2:((360 * betta)+d4/2) 
        angle3 = alpha * j ;
        column_mesh = 1001 + round(R0.*cosd(angle3));
        row_mesh = 1001 - round(R0.*sind(angle3));
        phy_matrix(row_mesh,column_mesh) = (gen_surf(j,R0));
        %disp(row_mesh);
    end
end


%%
 imageArray = phy_matrix;

% Identify the zero-intensity region
mask = (imageArray == 0);

% Get the coordinates of all pixels
[x, y] = meshgrid(1:size(imageArray, 2), 1:size(imageArray, 1));

% Get the coordinates and values of the non-zero intensity pixels
valid_x = x(~mask);
valid_y = y(~mask);
valid_values = double(imageArray(~mask));

% Create grid for interpolation
[xq, yq] = meshgrid(1:size(imageArray, 2), 1:size(imageArray, 1));

% Perform interpolation
interpolated_image = griddata(valid_x, valid_y, valid_values, xq, yq, 'cubic');

% Fill the zero-intensity region with the interpolated values
image_filled = double(imageArray);
image_filled(mask) = interpolated_image(mask);



%%
for R0=1:1000
    for j=d4/2:((360 * betta)+d4/2) 
        angle3 = alpha * j ;
        column_mesh = 1001 + round(R0.*cosd(angle3));
        row_mesh = 1001 - round(R0.*sind(angle3));
        phy_matrix(row_mesh,column_mesh) = (image_filled(row_mesh,column_mesh));
        %surface_fil(row_mesh,column_mesh) = surface(row_mesh,column_mesh);
        %disp(row_mesh);
    end
end

for R0=1:1000
    for j=d4/2:((360 * betta)+d4/2) 
        angle3 = alpha * j ;
        column_mesh = 1001 + round(R0.*cosd(angle3));
        row_mesh = 1001 - round(R0.*sind(angle3));
        %phy_matrix(row_mesh,column_mesh) = (image_filled(row_mesh,column_mesh));
        surface_fil(row_mesh,column_mesh) = surface(row_mesh,column_mesh);

    end 
end

surface_fil = surface_fil/max(max(surface_fil));
phy_matrix = phy_matrix/max(max(phy_matrix));

surface_fil(isnan(surface_fil)) = 0;
phy_matrix(isnan(phy_matrix)) = 0;

phy_matrix(phy_matrix<0)=0;

subplot(1,4,1), mesh(surface_fil);
subplot(1,4,2), mesh(phy_matrix);
subplot(1,4,3), imagesc(surface_fil);
subplot(1,4,4), imagesc(phy_matrix);


%%
% Assuming f and g are your 2D matrices
f = surface_fil;  % Replace with your actual matrix
g = phy_matrix;  % Replace with your actual matrix

% Check that f and g have the same size
if ~isequal(size(f), size(g))
    error('Matrices f and g must have the same size');
end

% Assuming x and y are the grid vectors
x = linspace(-7.005, 7.005, size(f, 1));
y = linspace(-7.005, 7.005, size(f, 2));

% Element-wise multiplication
product = f .* g;

% Numerical integration using trapz
% Integrate over y (rows) first
integral_y = trapz(y, product, 2);

% Integrate the result over x (columns)
integral = trapz(x, integral_y);

% M = intensity .* intensity_i;
% B = (trapz(x,trapz(x, abs(M), 1)) ).^2;
% C = trapz(x,trapz(x, abs(intensity).^2, 1)) * trapz(x,trapz(x, abs(intensity_i).^2, 1 ));
% OI = B./C;
% 
% disp(OI);

disp(integral);

%%
elapsedTime = toc;
disp(elapsedTime);
%disp(x);

%%
% Theta = linspace(0, 2*pi, size(C_ideal, 1));
% 
% [max_values, max_indices] = max(C_ideal);
% angles_max_intensity = Theta(max_indices);
% imagesc(angles_max_intensity);
% disp(angles_max_intensity);


%%
% mesh(C_ideal);



% imagesc(C);
%imagesc(C_ideal(:,500));
 % diff = C-C_ideal;
 % imagesc(R,theta,diff);


%%


 % mesh_matrix= zeros(1001,1001);
 % for R0 = 5:1000 % for vallues less than 10 the number of peaks is different
 %     C1 = C_fill_1(:,R0);
 %     C1_ideal = C_ideal_fill_1(:,R0);
 % 
 % 
 % 
 %     plot(C1);           % C1 = intensity vs angle at constant R
 %     hold on;
 %     plot(C1_ideal);
 %     hold off;
 % end
% 
% % % Dummy angle == angle at peaks for ideal superposition
%      dummy_angle_max = d1:d1:((360 * betta)+(d1));
%      max_angles_ideal = dummy_angle_max;   % define range
%       imagesc(max_angles_ideal);
% % % Maxima
%      max_angles_displaced(1:(2*l2 + 1)) = 0;
%      n0 = 1:d1:((360 * betta) + 1);           
%      m0 = d1:d1:((360 * betta)+(d1));
%      for i = 1: (2*l2 +1)
%          n1 = n0(i);
%          m1 = m0(i);
%          q1 = zeros(1,(360 * betta)+d1);               
%          q1(n1:m1) =  C(n1:m1,R0); 
%          % imagesc(q1);
%          [a1,b1] = max(q1);
%          max_angles_displaced(i) = b1;
%      end
% % %       max_angles_displaced;
% 
%      rotation_angles_1 = max_angles_ideal - max_angles_displaced ;
% 
% 
%      for j=1:(2*l2 + 1)
%          angle1 = max_angles_ideal(j) ./ betta;
%          coulmn_mesh = 501 + round(R0.*cosd(angle1));
%          row_mesh = 501 - round(R0.*sind(angle1));
%          mesh_matrix(row_mesh, coulmn_mesh) = (rotation_angles_1(j)) ./ betta;
%          %mesh(x,x,mesh_matrix);
%          %hold on;
%      end    % till here, peaks are stored in mesh matrix 
% 
% 
% % % Minima
%      min_angles_displaced(1:(2*l2)) = 0;
%      n2 = max_angles_displaced(1:2*l2);
%      m2 = max_angles_displaced(2:2*l2+1);
%      for i=1:2*l2
%          n21 = n2(i);
%          m21 = m2(i);
%          q2 = zeros(1,(360 * betta)+(d1));
%          q2(n21 : m21) = C1(n21 : m21);
%          %imagesc(q2);
%          i_max = zeros(1,(360 * betta)+(d1));
%          q3 = abs(q2 - i_max);
%          i_max(n21 : m21) = C1(max_angles_displaced(i));  % max intensity at ith peak
%          [a2,b2] = max(q3); 
%          min_angles_displaced(i) = b2;
% 
%      end
%      % imagesc(min_angles_displaced);
% % %        min_angles_displaced;
% 
% % % to store the elements in location matrix  
%      d2 = (360 * betta)/(4*l2);   %  it gives 1st minima for ideal; but in code we need minima 2nd to 2*l + 1
%      d1 = (360 * betta)/(2*l2);
%      min_angles_ideal = d2 : d1 : (360 * betta)+d1;
%      C4 = zeros(1,(360 * betta)+d1);
%      location_displaced = zeros(1,4*l2);
%      location_ideal = zeros(1,4*l2);
%      c = [min_angles_displaced max_angles_displaced];
%      % imagesc(c);
%      d = sort(c,"ascend");
%      % imagesc(d);
%      v1 = round((2.9 .* 360 * betta) / (2.*l2));
%      v2 = round((4.1 .* 360 * betta) / (2.*l2));
% 
%      for thetha = v1:v2    % angle for ideal pattern
%         % thetha = ((2 .* 360 * betta) / l):((3 .* 360 * betta) / l)            
%         i0 = C_ideal(thetha,R0);
%         for i=1:4*l2
%             %C4 = zeros(1,(360 * betta)+d1);
%             n3 = d(i);       
%             m3 = d(i+1);       
%             C4(n3:m3) = C((n3:m3),R0);
%             % imagesc(C4);
%            [a3,b3] = min(abs(C4 - i0));              
%             verify1 = C(b3,R0);
%            location_displaced(i) = b3;
%            imagesc(location_displaced);
%         end
%         %imagesc(location_displaced);
% % %           location_displaced;
% 
%         d_ideal = d1 : d2 : (360 * betta)+d1 ;
%         for i=1:4*l2
%             %C4_ideal = zeros(1,(360 * betta)+d1);
%             n4 = round(d_ideal(i));   
%             m4 = round(d_ideal(i+1));
%             C4_ideal(n4:m4) = C_ideal((n4:m4),R0);
%             %imagesc(C4_ideal);
%             [a4,b4] = min(abs(C4_ideal - i0));       
%             verify2 = C_ideal(b4,R0);
%             location_ideal(i) = b4;
%         end
%         %imagesc(location_ideal);
% % %           location_ideal;
% 
%         rotation_angles_2 = (location_ideal - location_displaced); 
% 
%        for j=1:(4*l2)
%             angle2 = location_ideal(j) ./ betta;
%             coulmn_mesh = 501 + round(R0.*cosd(angle2));         
%             row_mesh = 501 - round(R0.*sind(angle2));
%             mesh_matrix(row_mesh,coulmn_mesh) = (rotation_angles_2(j)) ./ betta;
%             % mesh(mesh_matrix);
%             % hold on;
%         end 
%     end  % loop on thetha
%  end % loop of R0
% 
%   mesh_matrix;
%   % mesh(x,x,mesh_matrix);


%%

% M = intensity .* intensity_i;
% B = (trapz(x,trapz(x, abs(M), 1)) ).^2;
% C = trapz(x,trapz(x, abs(intensity).^2, 1)) * trapz(x,trapz(x, abs(intensity_i).^2, 1 ));
% OI = B./C;
% 
% disp(OI);
%%
% M = norm_Ag2 .* norm_A2;
% B = (trapz(x,trapz(x, abs(M), 1)) ).^2;
% C = trapz(x,trapz(x, abs(norm_Ag2).^2, 1)) * trapz(x,trapz(x, abs(norm_A2).^2, 1 ));
% OI = B./C;
% disp(OI);

%%
% del_phy = k.*exp(-r.^2/3) ;
% % mesh(del_phy);
% % imagesc(del_phy);
% 
% phy_matrix= zeros(1001,1001);
% for R=1:500
%     for thetha=1:(360 * betta)
%         theta = alpha .* thetha;
%         rows = (501  -  round(R.*sind(theta)));
%         cols = (501  +  round(R.*cosd(theta)));
%         phy_1(thetha,R) = del_phy(rows,cols);
%     end
% end
% 
% for R0=5:500
%     for j=1:(360 * betta)
%         angle3 = alpha * j ;
%         coulmn_mesh = 501 + round(R0.*cosd(angle3));
%         row_mesh = 501 - round(R0.*sind(angle3));
%         phy_matrix(row_mesh,coulmn_mesh) = (phy_1(j,R0));
%     end
% end

 % mesh(phy_matrix)
% imagesc(phy_matrix)
%%
% aa1 = max(phy_matrix);
% original_max = max(aa1)
% 
% aa2 = max(mesh_matrix);
% obtained_max = max(aa2)
% 
% d3 = 360 / (2*l2);
% selected_mesh_matrix = abs(mesh_matrix) ;
% 
% for i = 1:1001
%     for j = 1:1001
%         if selected_mesh_matrix(i,j) > (original_max + (4/l2))
%            selected_mesh_matrix(i,j) = 0;
%         end
%     end
% end

%selected_mesh_matrix;

%mesh(abs(selected_mesh_matrix));





function v = LaguerreL(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Plot Generalized Legendre Polynomials: LaguerreL(n,z)
%
%               Coded by Manuel Diaz, NHRI, 2018.08.28.
%                   Copyright (c) 2018, Manuel Diaz.
%                           All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Refs.:
% [1] Milton Abramowitz, Irene Stegun, Handbook of Mathematical Functions,
%     National Bureau of Standards, 1964, ISBN: 0-486-61272-4, LC:QA47.A34.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluates the Generalized Laguerre polynomials GL(n,k,x).
%
%  First terms:
%
%    GL(0,k,x) = 1;
%    GL(1,k,x) = 1 + k - x;
%    GL(2,k,x) = 2 + 3*k + k^2 - 4*x - 2*k*x + x^2)/2;
%    GL(3,k,x) = 6 + 11*k + 6*k^2 + k^3 - 18*x - 15*k*x - 3*k^2*x + ...
%                  9*x^2 + 3*k*x^2 - x^3)/6.
%  Recursion:
%
%    GL(0,k,X) = 1 
%    GL(1,k,X) = 1 + k - x;
%
%    if 2 <= a:
%
%    GL(n,k,X) = ( (k+2*n-1-X) * GL(n-1,k,X) + (1-k-n) * GL(n-2,k,X) ) / n
%
%  Special values:
%
%    For k = 0, the associated Laguerre polynomials GL(N,K,X) are equal 
%    to the Laguerre polynomials L(N,X).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program and its subprograms may be freely used, modified and
% distributed under the GNU General Public License: 
% http://www.gnu.org/copyleft/gpl.html   
%
% Basic Assumptions:
% n: is a positive interger value.
% k: is a positive interger value.
% k: can be a single value or vector array.   
%
% Function inputs:
if (nargin == 2)     % Evaluate classical Laguerre Polynomials
    n=varargin{1};
    k=0;
    x=varargin{2};
elseif (nargin == 3) % Evaluate generalized Laguerre Polynomials
    n=varargin{1};
    k=varargin{2};
    x=varargin{3};
else 
    error('Usage: >> LaguerreL(n:int,k:int,x:array)');
end

% Verify inputs
if rem(n,1)~=0, error('n must be an integer.'); end
if rem(k,1)~=0, error('k must be an integer.'); end
if n < 0, error('n must be positive integer.'); end
if k < 0, error('k must be positive integer.'); end

% Initialize solution array
v = zeros(size(x));

% Compute Laguerre Polynomials
GL = zeros( numel(x), n+1 );
if n==0
    v(:) = 1.0;         % GL(0,k,x)
elseif n==1
    v(:) = 1+k-x(:);    % GL(1,k,x)
elseif n>1
    GL(:,1) = 1;        % GL(0,k,x)
    GL(:,2) = 1+k-x(:); % GL(1,k,x)
    for i = 2:n
        GL(:,i+1) = ( (k+2*i-1-x(:)).*GL(:,i) + (-k-i+1).*GL(:,i-1) )/i;
    end
    v(:) = GL(:,i+1);   % GL(n,k,x)
end
end




