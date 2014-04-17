void init_vars(double *rho, double *u, int *is_solid, mystruct *param)
{
  int ex[9] = { 0, 1, 0, -1,  0, 1, -1, -1,  1};
  int ey[9] = { 0, 0, 1,  0, -1, 1,  1, -1, -1};
  double wt[9]={4./9.,1./9.,1./9.,1./9.,1./9.,
                     1./36.,1./36.,1./36.,1./36.};
int i, j, idx, a;

if(param->comp==0)
{
for (i=0; i< param->N; i++)
 rho[i] = param->rho0_in;
}
else
{
  //rho[i] =(i < param->N/2) ? param->rho1_in+1 : param->rho1_in ;
  //rho[i] =(i%(param->LX/2)==0)? param->rho1_in+1 : param->rho1_in ;
for (j=0;j<param->LY;j++)
{
  for (i=0; i<param->LX; i++)
  {
	idx=j*param->LX+i;
    //if(pow((i-param->LX/2),2.)+pow((j-param->LY/2),2.) < pow(param->radii,2.)) //CIRCLE IN THE MIDDLE
	if(j==ceil(param->LY/2+4.*cos(10.*3.1414*i/param->LX)) ) 
	//if(j==ceil(1+4.*cos(10.*3.1414*i/param->LX)) ) 
	//if(j==5 && i==param->LX/2)
	//if(i>ceil(param->LX/2-4.*cos(2.*3.1414*j/param->LY)) ) 
	//if(i< param->LX-2 ) 
	//SINUSOIDAL INITIAL COND
    	{ rho[idx]=param->rho1_bcs + IC*param->dRho*sin(3.117*i/param->LX); }
    	else{rho[idx]=param->rho1_bcs  ; }
  }
}
}

# if PARABOLIC_ICS
	
    int mm=0, idx, j;
double *u_pois,  nu=(0.3333333)*(param->tau0-0.5);
  u_pois = (double *)malloc(param->LY*sizeof(double));

for(i=-(param->LY-2)/2;i<=(param->LY-2)/2; i++)
{
    u_pois[mm]=(param->gr_x/(2*nu))*(pow((param->LY-2)/2,2.)-pow(i,2.) );
    u_pois[0]=0.;
    mm=mm+1;
}
u_pois[mm]=0.;
for (j=0;j<param->LY;j++)
{
  for (i=0; i<param->LX; i++)
  {
	idx=j*param->LX+i;
   u[idx*DIM+0]= u_pois[j] ;
   u[idx*DIM+1]= param->uy_in ;
	
  }

}
free(u_pois);
#else
double uin[DIM]={param->ux_in, param->uy_in};//, param->uz_in};
for (i=0; i< param->N; i++)
{
  for(a=0; a<DIM; a++)
    u[i*DIM+a]= uin[a];
}

#endif
param->t=0;
param->ts = param->frame_rate * param->num_frame;

  for(a=0; a<Q; a++)
	param->F_gr[a] = 3.*wt[a]*( ex[a]*param->gr_x + ey[a]*param->gr_y);// +ez[a]*param->gr_z);

}
