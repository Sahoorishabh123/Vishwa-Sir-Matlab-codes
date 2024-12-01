

function LG_beam_superposition(P,L,A,W)


%% section 1
Grid= linspace(-5,4.98,500);
[X,Y] = meshgrid(Grid);
[phi,r] = cart2pol(X,Y);


%Laguerre-Gauss equaiton: 
t = (X.^2+Y.^2)/(W^2);



lambda = 633e-9;
k=2*pi/lambda;


  


Phi = L.*atan2(Y,X);
PhiM = (-L).*atan2(Y,X);

Term1 =((sqrt(2)*sqrt(X.^2 + Y.^2)/W)).^L;
Term1M =((sqrt(2)*sqrt(X.^2 + Y.^2)/W)).^(L);
Term2 = LaguerreL(500,0,2.*t);
Term3 = exp(-t);
Term4 = exp(1i*Phi);
Term4M = exp(1i*PhiM);
Z = A.*Term1.*Term2.*Term3.*Term4;

ZM= A.*Term1M.*Term2.*Term3.*Term4M;
S = Z+ZM;
%m=exp(1i*2*k*f);

%plot3(X,Y,f);


Spatial = real(S);
Phase = angle(S);

Intensity = abs(S);




%Plots
%% section 2
figure('Name',strcat('LG',num2str(P),',',num2str(L)),'Renderer', 'painters', 'Position', [125 125 1300 300])
subplot(1,3,1)
xlabel('E(x,y)')
xlim([1 length(Grid)]);ylim([1 length(Grid)]);
surface(Spatial)
colormap(jet)
colorbar()
shading interp

subplot(1,3,2)
xlabel('|E(x,y)|\^{2}')
xlim([1 length(Grid)]);ylim([1 length(Grid)]);
surface(Intensity)
colormap(hot)
colorbar()
shading interp 

subplot(1,3,3)
xlabel('Phase')
xlim([1 length(Grid)]);ylim([1 length(Grid)]);
surface(Phase)
colormap(hot)
colorbar('Ticks',[-pi -pi/2 0 pi/2 pi], 'Ticklabels',{'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
shading interp 


end


