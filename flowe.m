clear all;clc;
close all;

 lx=64;ly=lx;frame=10; ab=lx/2-4; out='out';
 gr=3.0e-06;beta=1.;dT=15.59;t=2500*frame;
   
   tau0=0.6; tau1=.51;%0.682;
   nu=0.333333*(tau0-0.5); k=0.333333*(tau1-0.5);
   Ra=gr*beta*dT*((ly-2)^3)/(nu*k);Pr=nu/k;
 for frame=1:20
 ux=load(sprintf('./%s/ux%dx%d_frame%03d.dat',out,lx,ly,frame));
 ux=reshape(ux,lx,ly); 
 uy=load(sprintf('./%s/uy%dx%d_frame%03d.dat',out,lx,ly,frame));
 uy=reshape(uy,lx,ly); 
  ux=ux'; 
uy=uy';


rho1=load(sprintf('./%s/rho[1]%dx%d_frame%03d.dat',out,lx,ly,frame));
rho1=reshape(rho1,lx,ly); rho1=rho1';
 figure;[c,h]=contourf(rho1);axis equal tight;
 set(h,'Edgecolor','none');colorbar;%colorbar('horiz');
% 
% title(['t=' num2str(t) '     Ra number is ' num2str(Ra) ]);
% figure;
% 
  sx1=1;sx2=lx; sy1=1; sy2=ly; 
  [x, y]=meshgrid(sx1:sx2,sy1:sy2); nx=ceil(lx/3);
%for j=3:19:ly
 %sx=1:3:lx; sy=j*ones(1,nx);
%  sx=[80 ]; sy=[15 ];
%  h=streamline(x,y,ux(sy1:sy2,sx1:sx2),uy(sy1:sy2,sx1:sx2),sx, sy);
%  set(h,'Color','red');
%  hold on;
%end

%  sy=1:4*lx/ly:ly; sx=1:(4*lx/ly):lx;
%  h=streamline(x,y,ux(sy1:sy2,sx1:sx2),uy(sy1:sy2,sx1:sx2),sx, sy);
%  
%  sy=ly:-4*lx/ly:1; sx=1:(4*lx/ly):lx;
%  h=streamline(x,y,ux(sy1:sy2,sx1:sx2),uy(sy1:sy2,sx1:sx2),sx, sy);
%  set(h,'Color','k');axis equal tight;
% 
% 
%  hold off;
 
 %figure
 %contourf(ux); axis equal; colorbar;
 Nu(frame,1)=0;
 for j=1:lx
    
 Nu(frame,1)=Nu(frame,1)+(rho1(1,j)-rho1(2,j))/dT; %Kuzmin bottom wall
 %Nu(frame,1)=Nu(frame,1)+(rho1(j,1)-rho1(j,2))/dT; %Kuzmin side wall
 end
   %  Nu(frame,1)    = 1. + sum(sum(ux.*rho1))/(ly*k*(dT)); %Tolke
 
 Nu(frame,1)
saveas(gcf,sprintf('./out/file%dx%d_frame%03d.png',lx,ly,frame),'png');
close all
 end 
 figure
 plot(Nu)
%  title(['t=' num2str(t) '    Ra = ' num2str(Ra) '  Pr='  num2str(Pr)])
%export_fig(sprintf('./%s/Ra=1e04_frame%03d', out,frame),'-m3', '-a1', '-pdf','-eps','-jpg');

 
 %axis([5 95 130 250]);
% u_pois=zeros(ly); tau0=1.;
% nu=(0.333333)*(tau0-0.5);
% ni=ly; g=1.e-05;
% dpdl=(1/3)*(8.e-04)/(lx-1);
% mm=1;
% 
% for i=-(ni-1)/2:(ni-1)/2
%     u_pois(mm)=(g/(2*nu))*(((ni-2)/2)^2-i^2);
%     %u_pois(mm)=(dpdl/(2*nu))*(((ni-2)/2)^2-i^2);
%     u_pois(1)=0.0;
%     mm=mm+1;
% end
% u_pois(mm)=0.0;
% hold on
% plot(u_pois(1:ly),1:ly,'r','LineWidth',2)
