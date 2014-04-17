void write_data(double *, double *, mystruct *);
void write_data(double *rho, double *u, mystruct *param)
{
  FILE *fluid_data, *ux_data, *uy_data; 
  char  fluid_file[1024], ux_file[1024], uy_file[1024];
  int i, N, LX, LY, frame;
LX= param->LX;LY= param->LY;frame= param->t/param->frame_rate;
param->frame= frame; N= LX*LY;

	if(param->comp==0)
	{
	  sprintf( fluid_file, "./out/rho[0]%dx%d_frame%03d.dat", LX,LY,frame );
	  sprintf(   ux_file , "./out/ux%dx%d_frame%03d.dat", LX,LY, frame );
	  sprintf(   uy_file , "./out/uy%dx%d_frame%03d.dat", LX,LY, frame );

	  if(!( fluid_data = fopen(fluid_file,"w+")) 
          || !( ux_data = fopen(ux_file,"w")) 
          || !( uy_data = fopen(uy_file,"w"))
            )
	 {
	 printf("\n(%d)---ERROR:---%s file missing\n", __LINE__, fluid_file);
	 exit(1);
	  }
	}
	FILE *sol_data; char sol_file[1024];
if(param->comp==1 && (param->t >= param->sigma_start || param->t==0) )
{
sprintf(   sol_file, "./out/rho[1]%dx%d_frame%03d.dat", LX,LY, frame );
 sol_data = fopen(sol_file,"w");
}
       // check results
          for (i=0; i<N; i++) 
          {
	if(param->comp==0)
	{
             fprintf(fluid_data,"%12.10lf\n", rho[i]);
             fprintf(ux_data,   "%12.10lf\n", u[i*DIM+0]);
             fprintf(uy_data,   "%12.10lf\n", u[i*DIM+1]);
	}
	if(param->comp==1 && (param->t >= param->sigma_start || param->t==0) )
             fprintf(sol_data,  "%12.10lf\n", rho[i]);
          }

if(param->comp==1 && (param->t >= param->sigma_start || param->t==0) )
fclose(sol_data); 
if(param->comp==0)
	{fclose(fluid_data); fclose(ux_data); fclose(uy_data);
    	printf("frame=%d frame_rate=%d t=%d\n", param->frame,param->frame_rate, param->t);

time_t result;
    result = time(NULL);
    printf("%s \n", asctime(localtime(&result)));
	}
}
