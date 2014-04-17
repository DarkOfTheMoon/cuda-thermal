__global__  void streaming(double *f0, double *ftemp0, mystruct *param)
{
  int ip, in, jp, jn, i5, i6, i7, i8;
  int N, LX, LY;
int idx=threadIdx.x + blockIdx.x * blockDim.x; 
if(idx<param->N)
{
  LX= param->LX; LY=param->LY; N= LX*LY;

    //----------------S T R E A M I N G----
 if(param->comp==0)
{
         ip= (idx+1)%LX==0 ? idx-(LX-1) : idx+1;
         in= (idx)%LX==0 ? idx+(LX-1) : idx-1;
         jp= idx<(N-LX) ? idx+LX : idx-N+LX;
         jn= idx<LX ? idx+N-LX : idx-LX;
         i5= (idx+1)%LX==0 ? idx+1 : idx+LX+1;
         if(idx>=N-LX){i5=idx-(N-LX)+1;} if(idx==N-1){i5=0;}
         i6= (idx)%LX==0 ? idx+2*LX-1 : idx+LX-1;
         if(idx==N-LX){i6=LX-1;} if(idx>N-LX){i6=idx-(N-LX)-1;}
         i7= (idx)%LX==0 ? idx-1 : idx-(LX+1);
         if(idx<=LX-1){i7=idx+(N-LX)-1;} if(idx==0){i7=N-1;}
         i8= (idx+1)%LX==0 ? idx-2*LX+1 : idx-(LX-1);
         if(idx==LX-1){i8=N-LX;} if(idx<LX-1){i8=idx+(N-LX)+1;}
         
          ftemp0[Q*ip +1] = f0[idx*Q +1] ; 
          ftemp0[Q*jp +2] = f0[idx*Q +2] ; 
          ftemp0[Q*in +3] = f0[idx*Q +3] ; 
          ftemp0[Q*jn +4] = f0[idx*Q +4] ; 
          ftemp0[Q*i5 +5] = f0[idx*Q +5] ; 
          ftemp0[Q*i6 +6] = f0[idx*Q +6] ; 
          ftemp0[Q*i7 +7] = f0[idx*Q +7] ; 
          ftemp0[Q*i8 +8] = f0[idx*Q +8] ; 
          ftemp0[Q*idx+0] = f0[idx*Q +0] ; 
}
else
{

         ip= (idx+1)%LX==0 ? idx-(LX-1) : idx+1;
         in= (idx)%LX==0 ? idx+(LX-1) : idx-1;
         jp= idx<(N-LX) ? idx+LX : idx-N+LX;
         jn= idx<LX ? idx+N-LX : idx-LX;

          ftemp0[5*ip +1] = f0[idx*5 +1] ; 
          ftemp0[5*jp +2] = f0[idx*5 +2] ; 
          ftemp0[5*in +3] = f0[idx*5 +3] ; 
          ftemp0[5*jn +4] = f0[idx*5 +4] ; 
          ftemp0[5*idx+0] = f0[idx*5 +0] ; 
}
}
}

__global__  void streamingUpdate(double *f0, double *ftemp0, mystruct *param)
{
  int a;
int idx=threadIdx.x + blockIdx.x * blockDim.x; 

if(idx<param->N)
{

if(param->comp==0)
{
          for(a=0; a<Q; a++)
          {
            f0[Q*idx+a] = ftemp0[Q*idx+a];   
          }
}
else
{
          for(a=0; a<5; a++)
          {
            f0[5*idx+a] = ftemp0[5*idx+a];   
          }

}


}
}
