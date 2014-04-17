__global__ void init_f(double *f, double *rho, double *u, mystruct *param )
{
int idx=threadIdx.x + blockIdx.x * blockDim.x,a ;

if(idx<param->N)
{
if(param->comp==0)
{
  double wt[Q]={4./9.,1./9.,1./9.,1./9.,1./9.,
                     1./36.,1./36.,1./36.,1./36.};
	for(a=0; a<Q; a++)
	f[idx*Q+a] = wt[a]*rho[idx];
}
else
{
double wt[5] = {1./3., 1./6., 1./6., 1./6., 1./6. }; 

	for(a=0; a<5; a++)
	f[idx*5+a] = wt[a]*rho[idx];
}
}
}
