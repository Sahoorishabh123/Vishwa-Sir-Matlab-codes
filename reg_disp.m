
function[reg_disp] = reg_disp(t,l1,n1,Z2,d)


x = linspace(-7.005,7.005,2001);
[X,Y] = meshgrid(x,x);
[phi,r] = cart2pol(X,Y);





threshold = 0.95; % 0.15 for the whole beam

alpha = 0.1;   % resolution factor
betta = 1 ./ alpha;
d1 = (360 .* betta)/(2*l1);
intensity_i_fill = beam_func_i_fill(r,Z2,phi,l1,n1,0.1,t);
intensity_fill = beam_func_fill(r,Z2,phi,l1,n1,0.1,t);

if l1 == 2
    m = 0.28411285;
    c = 0.0102449;

elseif l1 == 3
    m = 0.18935774;
    c = 0.1537551;

elseif l1 == 4
    m = 0.14193037;
    c =  0.01355102;

elseif l1 == 5
    m = 0.11356423;
    c = 0.01322449;

elseif l1 == 6
    m = 9.46854742e-02;
    c = 4.08163244e-05;

elseif l1 == 8
    m = 0.07077551;
    c = 0.04244898;


elseif l1 == 9
    m = 0.06287995;
    c = 0.02412245;

end






for R=1:1000
             for theta=d1/2:((360 * betta)+(d1/2))
               thetha = alpha .* theta;
               rows = (1001  -  round(R.*sind(thetha)));
               cols = (1001 +  round(R.*cosd(thetha)));
               
               C_fill(theta,R) = intensity_fill(rows,cols);
               C_ideal_fill(theta,R) = intensity_i_fill(rows,cols);
              

             end
end

        threshold = 0.80; % 0.15 for the whole beam

    
        C_ideal_fill(C_ideal_fill<threshold)  = 0.00;
        C_fill(C_fill<threshold)  = 0.00;
     
    
      % Find the columns (x-axis indices) with non-zero values
     nonZeroColumns = any(C_ideal_fill ~= 0, 1);

     % Get the range of x-axis indices with non-zero values
     nonZeroIndices = find(nonZeroColumns);
     r_min = min(nonZeroIndices);
     r_max = max(nonZeroIndices);
     

   reg_disp = zeros(3750,1000);

   s = round((r_max-r_min)/d);

   if t == 0

   %  for rad = r_min
   % 
   % if(rad+s)>1000
   %       break;
   % end 
   %disp(rad);

   
  

     for th = d1/2:d1:((360 * betta)+(d1/2)) 

       if(th+(d1))>((360 * betta)+(d1/2))
            break;
       end 

       desired_part_ideal = C_ideal_fill(th:th+(d1),r_min:r_max);
       desired_part = C_fill(th:th+(d1),r_min:r_max);
       % diff = (desired_part-desired_part_ideal);
       maximum = max(max(desired_part));
       [x,y]=find(desired_part == maximum);
       data = [x,y];
       first_col = data(:,1);
       median_data = median(first_col);

       maximum_i = max(max(desired_part_ideal));
       [x_i,y_i]=find(desired_part_ideal == maximum_i);
       data_i = [x_i,y_i];
       first_col_i = data_i(:,1);
       median_data_i = median(first_col_i);
       shift = abs(median_data - median_data_i)/10;
        %disp(shift);
        %disp(median_data);
       
         if shift ~= 0

          reg_disp(th+(d1/2)-50,round((r_min+r_max)/2)) = ((shift-c)/m).*10^-9;

         else

          reg_disp(th+(d1/2)-50,round((r_min+r_max)/2)) = 0;
         end
       %disp(th);
     
      %imagesc(reg_disp);

    

    
        
    
     end
    end
   
% for t ~= 0

   % else
   % 
   %     for rad = r_min:s:r_max
   % 
   % if(rad+s)>1000
   %       break;
   % end 
   %disp(rad);

   
  

     for th = d1/2+d1-(1800*t):d1:((360 * betta)+(d1/2)-(1800*t)) 

       if(th+(d1))>((360 * betta)+(d1/2)-(1800*t))
            break;
       end 

       desired_part_ideal = C_ideal_fill(th:th+(d1),r_min:r_max);
       desired_part = C_fill(th:th+(d1),r_min:r_max);
       % diff = (desired_part-desired_part_ideal);
       maximum = max(max(desired_part));
       [x,y]=find(desired_part == maximum);
       data = [x,y];
       first_col = data(:,1);
       median_data = median(first_col);

       maximum_i = max(max(desired_part_ideal));
       [x_i,y_i]=find(desired_part_ideal == maximum_i);
       data_i = [x_i,y_i];
       first_col_i = data_i(:,1);
       median_data_i = median(first_col_i);
       shift = abs(median_data - median_data_i)/10;
        %disp(shift);
        %disp(median_data);
       
         if shift ~= 0

          reg_disp(th+(d1/2)-50,round((r_min+r_max)/2)) = ((shift-c)/m).*10^-9;

         else

          reg_disp(th+(d1/2)-50,round((r_min+r_max)/2)) = 0;
         end
       %disp(th);
     
      %imagesc(reg_disp);

    

    
        
    
     end
       end

   








  


