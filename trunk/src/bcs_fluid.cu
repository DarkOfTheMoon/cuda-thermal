__global__  void bcs_fluid(double *f0, mystruct *param )
{
int idx=threadIdx.x + blockIdx.x * blockDim.x ;
  double c, u ;
if(idx <param->N)
{
if(param->pressure_bcs_ew==1 && idx%param->LX==0 )//WEST 
{

     u =    1.//rho0_in
            - (      *(f0 + idx*9 + 0) + *(f0 + idx*9 + 2) + *(f0 + idx*9 +4)
              + 2.*( *(f0 + idx*9 + 3) + *(f0 + idx*9 + 7) + *(f0 + idx*9 +6) ) )/param->rho0_in;
        c = u*param->rho0_in;
      *(f0 + idx*9 + 1) = *(f0 + idx*9+ 3) + (2./3.)*c;

      *(f0 + idx*9 + 5) = *(f0 + idx*9+ 7) + (1./2.)*( *(f0 + idx*9 + 4) - *(f0 + idx*9  + 2))
                     + (1./6.)*c;

      *(f0 + idx*9 + 8) = *(f0 + idx*9+ 6) + (1./2.)*( *(f0 + idx*9  + 2) - *(f0 + idx*9 + 4))
                    + (1./6.)*c;
}
if( param->pressure_bcs_ew==1 && (idx+1)%param->LX==0)  //EAST
{

     u =    -1.//rho0_in
            + (      *(f0 + idx*9 + 0) + *(f0 + idx*9 + 2) + *(f0 + idx*9 +4)
              + 2.*( *(f0 + idx*9 + 1) + *(f0 + idx*9 + 5) + *(f0 + idx*9 +8) ) )/param->rho0_out;
        c = u*param->rho0_out;
      *(f0 + idx*9 + 3) = *(f0 + idx*9+ 1) - (2./3.)*c;

      *(f0 + idx*9 + 7) = *(f0 + idx*9+ 5) + (1./2.)*( *(f0 + idx*9 + 2) - *(f0 + idx*9 + 4))
                    - (1./6.)*c;

      *(f0 + idx*9 + 6) = *(f0 + idx*9+ 8) + (1./2.)*( *(f0 + idx*9 + 4) - *(f0 + idx*9 + 2))
                    - (1./6.)*c;

}
# if VELOCITY_BCS_EW
int idx=threadIdx.x + blockIdx.x * blockDim.x, a ;
  double ux_in, ux_out, c, rho;
if(idx%param->LX==0) //WEST
{
ux_in=param->ux_in;

     rho =    (      *(f0 + idx*9 + 0) + *(f0 + idx*9 + 2) + *(f0 + idx*9 +4)
              + 2.*( *(f0 + idx*9 + 3) + *(f0 + idx*9 + 7) + *(f0 + idx*9 +6) ) )/(1.-ux_in);
        c = ux_in*rho;
      *(f0 + idx*9 + 1) = *(f0 + idx*9+ 3) + (2./3.)*c;

      *(f0 + idx*9 + 5) = *(f0 + idx*9+ 7) + (1./2.)*( *(f0 + idx*9 + 4) - *(f0 + idx*9  + 2))
                     + (1./6.)*c;

      *(f0 + idx*9 + 8) = *(f0 + idx*9+ 6) + (1./2.)*( *(f0 + idx*9  + 2) - *(f0 + idx*9 + 4))
                    + (1./6.)*c;
}
if( (idx+1)%param->LX==0) //EAST
{
ux_out=param->ux_out;
     rho =  (          *(f0 + idx*9 + 0) + *(f0 + idx*9 + 2) + *(f0 + idx*9 +4)
              + 2.*( *(f0 + idx*9 + 1) + *(f0 + idx*9 + 5) + *(f0 + idx*9 +8) ) )/(1.+ux_out);
        c = ux_out*rho;
      *(f0 + idx*9 + 3) = *(f0 + idx*9+ 1) - (2./3.)*c;

      *(f0 + idx*9 + 7) = *(f0 + idx*9+ 5) + (1./2.)*( *(f0 + idx*9 + 2) - *(f0 + idx*9 + 4))
                    - (1./6.)*c;

      *(f0 + idx*9 + 6) = *(f0 + idx*9+ 8) + (1./2.)*( *(f0 + idx*9 + 4) - *(f0 + idx*9 + 2))
                    - (1./6.)*c;
}
#endif

}

}

