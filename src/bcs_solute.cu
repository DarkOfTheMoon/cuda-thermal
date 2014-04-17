__global__  void bcs_solute(double *f1, int * is_solid, mystruct *param)
{
int idx=threadIdx.x + blockIdx.x * blockDim.x, a;
if(idx <param->N)
{
if(param->solute_bcs_w==1 && idx%param->LX==0 )//WEST 
    f1[idx*5 + 1] =  param->rho1_bcs+ param->dRho - f1[ idx*5 + 0] - f1[ idx*5 + 2] - f1[idx*5 + 3] - f1[idx*5 + 4];

if(param->solute_bcs_e==1 &&  (idx+1)%param->LX==0)//EAST
    f1[idx*5 + 3] =  param->rho1_bcs - f1[ idx*5 + 0] - f1[ idx*5 + 1] - f1[idx*5 + 2] - f1[idx*5 + 4];


if(param->solute_bcs_s==1 && idx<param->N/2  && is_solid[idx] )//SOUTH 

    f1[idx*5 + 2] =  param->rho1_bcs+ param->dRho - f1[ idx*5 + 0] - f1[ idx*5 + 1] - f1[idx*5 + 3] - f1[idx*5 + 4];

if(param->solute_bcs_n==1 && idx>=param->N/2 && is_solid[idx] )//NORTH 
//if(param->solute_bcs_n==1 && idx>=param->N-param->LX && is_solid[idx] )//NORTH 

    f1[idx*5 + 4] =  param->rho1_bcs  - f1[ idx*5 + 0] - f1[ idx*5 + 1] - f1[idx*5 + 2] - f1[idx*5 + 3];

if(param->solute_bcs_zerograd_n ==1 && idx>=param->N-param->LX )// NORTH
{
	for(a=0; a<5; a++)
	f1[idx*5+a] = f1[(idx-param->LX)*5+a];
}
if(param->solute_bcs_zerograd_s ==1 && idx<param->LX )// SOUTH
{
	for(a=0; a<5; a++)
	f1[idx*5+a] = f1[(idx+param->LX)*5+a];
}
if(param->solute_bcs_zerograd_e ==1 && idx%param->LX ==0)// EAST
{
	for(a=0; a<5; a++)
	f1[idx*5+a] = f1[(idx+1)*5+a];
}
if(param->solute_bcs_zerograd_w ==1 && (idx+1)%param->LX ==0)// WEST
{
	for(a=0; a<5; a++)
	f1[idx*5+a] = f1[(idx-1)*5+a];
}
}
}

