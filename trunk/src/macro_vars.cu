__global__ void macro_vars(double *f, double *rho, double *u, mystruct *param )
{
  int ex[Q] = { 0, 1, 0, -1,  0, 1, -1, -1,  1};
  int ey[Q] = { 0, 0, 1,  0, -1, 1,  1, -1, -1};
  int idx=threadIdx.x + blockIdx.x * blockDim.x, a;
if(idx<param->N)
{
   //MACROSCOPIC VARIABLES
     rho[idx]=0.;
if(param->comp==0)
{
	u[idx*DIM+0]=0.; u[idx*DIM+1]=0.;
  
      for(a=0; a<Q; a++)
      {
              rho[idx] += f[idx*Q + a];
           u[idx*DIM+0]+= f[idx*Q+a]*ex[a];
           u[idx*DIM+1]+= f[idx*Q+a]*ey[a];
      }
      u[idx*DIM+0] = u[idx*DIM+0]/rho[idx];
      u[idx*DIM+1] = u[idx*DIM+1]/rho[idx];
  
}
else
{
      for(a=0; a<5; a++)
              rho[idx] += f[idx*5 + a];

}
}
}
