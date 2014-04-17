__global__ void collision(double *f, double *rho, double *rho1, double *u, int *is_solid, mystruct *param)
{
  int idx=threadIdx.x + blockIdx.x * blockDim.x, comp=param->comp, a;
  double temp, mean_rho, usq, edotu, edotu_sq, feq;
if(idx<param->N)
{
if(comp==0)
{
  int ex[9] = { 0, 1, 0, -1,  0, 1, -1, -1,  1};
  int ey[9] = { 0, 0, 1,  0, -1, 1,  1, -1, -1};
  double wt[9]={4./9.,1./9.,1./9.,1./9.,1./9.,
                     1./36.,1./36.,1./36.,1./36.}, ueq_x, ueq_y;
	mean_rho=param->rho1_bcs+param->dRho/2.;
   if(is_solid[idx] >0 )//---B O U N C E-B A C K--- C O L L I S I O N-----
   {
     temp = f[Q*idx+1]; f[Q*idx+1] = f[Q*idx+3]; f[Q*idx+3] = temp;
     temp = f[Q*idx+2]; f[Q*idx+2] = f[Q*idx+4]; f[Q*idx+4] = temp;
     temp = f[Q*idx+5]; f[Q*idx+5] = f[Q*idx+7]; f[Q*idx+7] = temp;
     temp = f[Q*idx+6]; f[Q*idx+6] = f[Q*idx+8]; f[Q*idx+8] = temp;
   }
   else
   {
	ueq_x= u[idx*DIM+0];// + param->gr_x*param->tau0*(rho1[idx]-mean_rho)*param->alpha;///param->dRho;
	ueq_y= u[idx*DIM+1];// + param->gr_y*param->tau0*(rho1[idx]-mean_rho)*param->alpha;///param->dRho;
    	usq = ueq_x*ueq_x + ueq_y*ueq_y;
    	//usq = u[idx*DIM+0]*u[idx*DIM+0] + u[idx*DIM+1]*u[idx*DIM+1];
	for(a=0; a<Q; a++)
	{
    	edotu = ex[a]*ueq_x + ey[a]*ueq_y;
    	//edotu = ex[a]*u[idx*DIM+0] + ey[a]*u[idx*DIM+1];
    	edotu_sq = edotu*edotu; 
	feq= rho[idx]*wt[a]*(1. + 3.*edotu + 4.5*edotu_sq - 1.5*usq);
	f[idx*Q+a] = f[idx*Q+a] - ( f[idx*Q+a] - feq)/param->tau0;
        f[idx*Q+a] +=  rho[idx]*param->F_gr[a]*param->alpha*(rho1[idx]- mean_rho);///param->dRho ;
	}

   }
}
else
{
int ex[5] = {  0,   1,   0,  -1,   0};
int ey[5] = {  0,   0,   1,   0,  -1};
double wt[5] = {1./3., 1./6., 1./6., 1./6., 1./6. };
   if(is_solid[idx] >1 )//---B O U N C E-B A C K--- C O L L I S I O N-----
   {
     temp = f[Q*idx+1]; f[Q*idx+1] = f[Q*idx+3]; f[Q*idx+3] = temp;
     temp = f[Q*idx+2]; f[Q*idx+2] = f[Q*idx+4]; f[Q*idx+4] = temp;
   }
   else
   {
	for(a=0; a<5; a++)
	{
    	edotu = ex[a]*u[idx*DIM+0] + ey[a]*u[idx*DIM+1];
	feq= rho[idx]*wt[a]*(1. + 3.*edotu );
	f[idx*5+a] = f[idx*5+a] - ( f[idx*5+a] - feq)/param->tau1_xx;
	}
   }

}
}
}
