#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "flags.h"

int main(int argc, char **argv)
{
	int device = 0; cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	cudaSetDevice( device );

mystruct *arg_ptr, arg, *arg_d ;arg_ptr= &arg;
read_params(arg_ptr);  dump_params(arg_ptr);
	printf("GPU device %d, name = %s\n", device, deviceProp.name );

	double *rho, *rho1, *u;
	double *f_d, *f1_d, *ftemp_d, *ftemp1_d, *rho_d, *rho1_d, *u_d;
	int *is_solid, *is_solid_d, t=0;
	size_t   size_f = arg_ptr->N*Q*sizeof(double);
	size_t     size = arg_ptr->N*sizeof(double);
	size_t size_int = arg_ptr->N*sizeof(int);

rho = (double *)malloc(size); u = (double *)malloc(DIM*size);
is_solid = (int *)malloc(size_int);   
rho1 = (double *)malloc(size); 

cudaMalloc((void **) &rho_d, size); cudaMalloc((void **)&u_d, DIM*size);
cudaMalloc((void **) &rho1_d, size); 
cudaMalloc((void **) &is_solid_d, size_int); 
cudaMalloc((void **) &f_d, size_f); cudaMalloc((void **)&ftemp_d, size_f);
cudaMalloc((void **) &f1_d, arg_ptr->N*5*sizeof(double)); 
cudaMalloc((void **) &ftemp1_d, arg_ptr->N*5*sizeof(double)); 
cudaMalloc(( void **) & arg_d, sizeof(mystruct ));
size_t freeMem = 0;
size_t totalMem = 0;
cudaMemGetInfo(&freeMem, &totalMem);  
printf("GPU Memory avaliable: Free: %lf GB, Total: %lf GB\n",freeMem/1e09, totalMem/1e09); 

size_f = arg_ptr->N*Q*sizeof(double);
printf("\nmem_reqd on host =%lf MB \n",((DIM+1)*size+size_int+2*Q*Q*4)/1.e06);
printf("\nmem_reqd on GPU  =%lf MB \n",(3*size_f+ (DIM+1)*size+size_int+2*Q*Q*4)/1.e06);

if(f_d ==NULL || ftemp_d ==NULL || u_d==NULL || rho_d==NULL || is_solid_d ==NULL || 
u==NULL || rho==NULL || is_solid ==NULL )
{printf("Memory allocation failed\n EXITING\n"); exit(1);}

	read_raw(is_solid, arg_ptr);
	arg_ptr->comp=0;
	init_vars(rho, u, is_solid, arg_ptr);
  	write_data(rho, u, arg_ptr);
	bitmap(arg_ptr->LX, arg_ptr->LY, 0, 1, 0, is_solid, rho) ; 
cudaMemcpy(arg_d, &arg, sizeof(mystruct), cudaMemcpyHostToDevice);
cudaMemcpy(rho_d, rho, (size), cudaMemcpyHostToDevice);
cudaMemcpy( u_d,  u, (DIM*size), cudaMemcpyHostToDevice);
cudaMemcpy(is_solid_d, is_solid, (size_int), cudaMemcpyHostToDevice);

	init_f<<< (arg_ptr->N+BL-1)/BL,BL  >>> (f_d, rho_d, u_d, arg_d);

	arg_ptr->comp=1;
	init_vars(rho1, u, is_solid, arg_ptr);
  	write_data(rho1, u, arg_ptr);
	bitmap(arg_ptr->LX, arg_ptr->LY, 0, 1, 1, is_solid, rho1) ; 
cudaMemcpy(arg_d, &arg, sizeof(mystruct), cudaMemcpyHostToDevice);
cudaMemcpy(rho1_d, rho1, (size), cudaMemcpyHostToDevice);
	init_f<<< (arg_ptr->N+BL-1)/BL,BL  >>> (f1_d, rho1_d, u_d, arg_d);

for (t=1; t<= arg_ptr->ts; t++)
{
  	arg_ptr->t = t; arg_ptr->comp=0;
 	cudaMemcpy(arg_d, &arg, sizeof(mystruct), cudaMemcpyHostToDevice);

  	streaming<<< (arg_ptr->N+BL-1)/BL, BL >>> (f_d, ftemp_d, arg_d);
  	streamingUpdate<<< (arg_ptr->N+BL-1)/BL, BL >>> (f_d, ftemp_d, arg_d);
  	bcs_fluid<<< (arg_ptr->N+BL-1)/BL, BL >>> (f_d, arg_d);
  	macro_vars<<< (arg_ptr->N+BL-1)/BL, BL >>> (f_d, rho_d, u_d, arg_d);
  	collision<<< (arg_ptr->N+BL-1)/BL, BL >>> (f_d, rho_d, rho1_d, u_d, is_solid_d, arg_d);

  	if(arg_ptr->t%arg_ptr->frame_rate==0)
  	{
cudaMemcpy(rho, rho_d, (size), cudaMemcpyDeviceToHost);
cudaMemcpy(  u,   u_d, DIM*(size), cudaMemcpyDeviceToHost);
write_data(rho, u, arg_ptr);
bitmap(arg_ptr->LX, arg_ptr->LY, t/arg_ptr->frame_rate, 1, 0, is_solid, rho) ; 
	}


  	arg_ptr->comp=1;
 	cudaMemcpy(arg_d, &arg, sizeof(mystruct), cudaMemcpyHostToDevice);

  	streaming<<< (arg_ptr->N+BL-1)/BL, BL >>> (f1_d, ftemp1_d, arg_d);
  	streamingUpdate<<< (arg_ptr->N+BL-1)/BL, BL >>> (f1_d, ftemp1_d, arg_d);
  	bcs_solute<<< (arg_ptr->N+BL-1)/BL, BL >>> (f1_d, is_solid_d, arg_d);
  	macro_vars<<< (arg_ptr->N+BL-1)/BL, BL >>> (f1_d, rho1_d, u_d, arg_d);
  	 collision<<< (arg_ptr->N+BL-1)/BL, BL >>> (f1_d, rho1_d, rho_d, u_d, is_solid_d, arg_d);

  	if(arg_ptr->t%arg_ptr->frame_rate==0)
  	{
cudaMemcpy(rho1, rho1_d, (size), cudaMemcpyDeviceToHost);
write_data(rho1, u, arg_ptr);
bitmap(arg_ptr->LX, arg_ptr->LY, t/arg_ptr->frame_rate, 1, 1, is_solid, rho1); 
	}

}
	free(rho); free(rho1); free(u); free(is_solid);
        cudaFree(is_solid_d); cudaFree(f_d); cudaFree(ftemp_d);
	cudaFree(f1_d); cudaFree(ftemp1_d);
	cudaFree(rho1_d); cudaFree(u_d); 
	return 0;
}
