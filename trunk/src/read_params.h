//##############################################################################
//Copyright: Dr. Danny Thorne and Dr. Michael Sukop
// R E A D   P A R A M   L A B E L 
//
//  - Read parameter label from parameter input file.
//
//  - Returns non-zero if label successfully read.
//
//  - Returns zero if end of file.
//
//  - Aborts with error message otherwise.
//
int read_param_label( FILE *in, char *param_label, int max_length)
{
  char c;
  int  i;

  if( max_length<=0)
  {
    printf("%s %d >> ERROR: max_length<=0 .\n",__FILE__,__LINE__);
    exit(1);
  }

  c = fgetc( in);

  if( feof(in)) { return 0;}

  if( ( c >= '0' && c <='9') || c == '-' || c == '.')
  {
    // Digit or Sign or Decimal. Labels cannot begin with these.
    // TODO: Check for other illegal initial characters.
    printf("%s %d >> Error reading label from parameters input file. "
        "Exiting!\n",__FILE__,__LINE__);
    fclose(in);
    exit(1);

  } /* if( ( c >= '0' && c <='9') || c == '-' || c == '.') */

  else
  {
    i = 0;
    param_label[i] = c;

    // Read label consisting of a string of non-whitespace characters.
    while( ( c = fgetc( in)) != ' ')
    {
      if( feof(in)) { return 0;}

      i++;
      if( i+1 > max_length)
      {
        printf("%s %d >> ERROR: i+1 > max_length .\n",__FILE__,__LINE__);
        exit(1);
      }
      param_label[i] = c;
    }

    // Terminate the string.
    i++;
    param_label[i] = '\0'; //(char)NULL;

    // Discard whitespace between label and its value. Calling routine
    // will read the value.
    while( ( c = fgetc( in)) == ' ')
    {
      if( feof(in)) { return 0;}
    }
    ungetc( c, in);

    //printf("%s %d >> Read label \"%s\"\n",__FILE__,__LINE__,param_label);

    return 1;

  } /* if( ( c >= '0' && c <='9') || c == '-' || c == '.') else */

} /* void read_param_label( FILE *in, char *param_label, int ... ) */

// void read_params( struct lattice_struct *lattice)
//##############################################################################
//
// R E A D   P A R A M S 
//
//  - Read the problem parameters from a file.
//
void read_params(  mystruct *param)
{
  FILE   *in;
  char   param_label[81], infile[1024];

	sprintf( infile, "./in/params.in" );

  if( !( in = fopen( infile, "r")))
  {
    printf("ERROR: fopen(\"%s\",\"r\") = NULL.  Bye, bye!\n", infile);
    exit(1);
  }


  // First assign default values to the parameters since this new mechanism
  // doesn't require all parameters to be specified.

  while( read_param_label( in, param_label, 80))
  {
    if( !strncmp(param_label,"LX",80))
    {
      fscanf( in, "%d\n", &(param->LX));
      printf("%s %d >> LX = %d\n",__FILE__,__LINE__, param->LX);
    }
    else if( !strncmp(param_label,"LY",80))
    {
      fscanf( in, "%d\n", &(param->LY));
      printf("%s %d >> LY = %d\n",__FILE__,__LINE__, param->LY);
    }
    else if( !strncmp(param_label,"num_frame",80))
    {
      fscanf( in, "%d\n", &(param->num_frame));
      printf("%s %d >> num_frame = %d\n",__FILE__,__LINE__, param->num_frame);
    }
    else if( !strncmp(param_label,"frame_rate",80))
    {
      fscanf( in, "%d\n", &(param->frame_rate));
      printf("%s %d >> frame_rate = %d\n",__FILE__,__LINE__, param->frame_rate);
    }
    else if( !strncmp(param_label,"tau0",80))
    {
      fscanf( in, "%lf\n", &(param->tau0));
      printf("%s %d >> tau0 = %f\n",__FILE__,__LINE__, param->tau0);
    }
    else if( !strncmp(param_label,"tau1_xx",80))
    {
      fscanf( in, "%lf\n", &(param->tau1_xx));
      printf("%s %d >> tau1_xx = %f\n",__FILE__,__LINE__, param->tau1_xx);
    }
    else if( !strncmp(param_label,"tau1_yy",80))
    {
      fscanf( in, "%lf\n", &(param->tau1_yy));
      printf("%s %d >> tau1_yy = %f\n",__FILE__,__LINE__, param->tau1_yy);
    }
    else if( !strncmp(param_label,"gr_x",80))
    {
      fscanf( in, "%le\n", &(param->gr_x));
      printf("%s %d >> gr_x = %e\n",__FILE__,__LINE__, param->gr_x);
    }
    else if( !strncmp(param_label,"gr_y",80))
    {
      fscanf( in, "%le\n", &(param->gr_y));
      printf("%s %d >> gr_y = %e\n",__FILE__,__LINE__, param->gr_y);
    }
    else if( !strncmp(param_label,"gr_b",80))
    {
      fscanf( in, "%le\n", &(param->gr_b));
      printf("%s %d >> gr_b = %e\n",__FILE__,__LINE__, param->gr_b);
    }
    else if( !strncmp(param_label,"ux_in",80))
    {
      fscanf( in, "%lf\n", &(param->ux_in));
      printf("%s %d >> ux_in = %f\n",__FILE__,__LINE__, param->ux_in);
    }
    else if( !strncmp(param_label,"uy_in",80))
    {
      fscanf( in, "%lf\n", &(param->uy_in));
      printf("%s %d >> uy_in = %f\n",__FILE__,__LINE__, param->uy_in);
    }
    else if( !strncmp(param_label,"uz_in",80))
    {
      fscanf( in, "%lf\n", &(param->uz_in));
      printf("%s %d >> uz_in = %f\n",__FILE__,__LINE__, param->uz_in);
    }
    else if( !strncmp(param_label,"rho0_in",80))
    {
      fscanf( in, "%lf\n", &(param->rho0_in));
      printf("%s %d >> rho0_in = %f\n",__FILE__,__LINE__, param->rho0_in);
    }
    else if( !strncmp(param_label,"rho0_out",80))
    {
      fscanf( in, "%lf\n", &(param->rho0_out));
      printf("%s %d >> rho0_out = %f\n",__FILE__,__LINE__, param->rho0_out);
    }
    else if( !strncmp(param_label,"rho1_in",80))
    {
      fscanf( in, "%lf\n", &(param->rho1_in));
      printf("%s %d >> rho1_in = %f\n",__FILE__,__LINE__, param->rho1_in);
    }
    else if( !strncmp(param_label,"rho1_bcs",80))
    {
      fscanf( in, "%lf\n", &(param->rho1_bcs));
      printf("%s %d >> rho1_bcs = %f\n",__FILE__,__LINE__, param->rho1_bcs);
    }
    else if( !strncmp(param_label,"sigma_start",80))
    {
      fscanf( in, "%d\n", &(param->sigma_start));
      printf("%s %d >> sigma_start = %d\n",__FILE__,__LINE__, param->sigma_start);
    }
    else if( !strncmp(param_label,"btc_pts",80))
    {
      fscanf( in, "%d\n", &(param->btc_pts));
      printf("%s %d >> btc_pts = %d\n",__FILE__,__LINE__, param->btc_pts);
    }
    else if( !strncmp(param_label,"btc_spot",80))
    {
      fscanf( in, "%d\n", &(param->btc_spot));
      printf("%s %d >> btc_spot = %d\n",__FILE__,__LINE__, param->btc_spot);
    }
    else if( !strncmp(param_label,"dRho",80))
    {
      fscanf( in, "%lf\n", &(param->dRho));
      printf("%s %d >> rho0_out = %lf\n",__FILE__,__LINE__, param->dRho);
    }
    else if( !strncmp(param_label,"radii",80))
    {
      fscanf( in, "%d\n", &(param->radii));
      printf("%s %d >> radii = %d\n",__FILE__,__LINE__, param->radii);
    }
    else if( !strncmp(param_label,"sigma",80))
    {
      fscanf( in, "%lf\n", &(param->sigma));
      printf("%s %d >> sigma = %f\n",__FILE__,__LINE__, param->sigma);
    }
    else if ( !strncmp(param_label,"alpha",80))
    {
      fscanf( in, "%lf\n", &(param->alpha));
      printf("%s %d >> alpha = %f\n",__FILE__,__LINE__, param->alpha);
    }
    else if( !strncmp(param_label,"pressure_bcs_ew",80))
    {
      fscanf( in, "%d\n", &(param->pressure_bcs_ew));
      printf("%s %d >> pressure_bcs_ew = %d\n",__FILE__,__LINE__, param->pressure_bcs_ew);
    }
    else if( !strncmp(param_label,"pressure_bcs_ns",80))
    {
      fscanf( in, "%d\n", &(param->pressure_bcs_ns));
      printf("%s %d >> pressure_bcs_ns = %d\n",__FILE__,__LINE__, param->pressure_bcs_ns);
    }
    else if( !strncmp(param_label,"solute_bcs_e",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_e));
      printf("%s %d >> solute_bcs_e= %d\n",__FILE__,__LINE__, param->solute_bcs_e);
    }
    else if( !strncmp(param_label,"solute_bcs_w",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_w));
      printf("%s %d >> solute_bcs_w= %d\n",__FILE__,__LINE__, param->solute_bcs_w);
    }
    else if( !strncmp(param_label,"solute_bcs_zerograd_n",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_zerograd_n));
      printf("%s %d >> solute_bcs_zerograd_n= %d\n",__FILE__,__LINE__, param->solute_bcs_zerograd_n);
    }
    else if( !strncmp(param_label,"solute_bcs_zerograd_s",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_zerograd_s));
      printf("%s %d >> solute_bcs_zerograd_s= %d\n",__FILE__,__LINE__, param->solute_bcs_zerograd_s);
    }
    else if( !strncmp(param_label,"solute_bcs_zerograd_e",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_zerograd_e));
      printf("%s %d >> solute_bcs_zerograd_e= %d\n",__FILE__,__LINE__, param->solute_bcs_zerograd_e);
    }
    else if( !strncmp(param_label,"solute_bcs_zerograd_w",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_zerograd_w));
      printf("%s %d >> solute_bcs_zerograd_w= %d\n",__FILE__,__LINE__, param->solute_bcs_zerograd_w);
    }
    else if( !strncmp(param_label,"solute_bcs_n",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_n));
      printf("%s %d >> solute_bcs_n= %d\n",__FILE__,__LINE__, param->solute_bcs_n);
    }
    else if( !strncmp(param_label,"solute_bcs_s",80))
    {
      fscanf( in, "%d\n", &(param->solute_bcs_s));
      printf("%s %d >> solute_bcs_s= %d\n",__FILE__,__LINE__, param->solute_bcs_s);
    }
    else if( !strncmp(param_label,"restart_frame_num",80))
    {
      fscanf( in, "%d\n", &(param->restart_frame_num));
      printf("%s %d >> restart_frame_num= %d\n",__FILE__,__LINE__, param->restart_frame_num);
    }
    else
    {
      printf("%s %d >> WARNING: Unhandled parameter \"%s\".\n",
        __FILE__,__LINE__,param_label);
    //  skip_rest_of_line( in);
    }

  }

  fclose(in);


} /* void read_params( struct lattice_struct *lattice) */

// void dump_params( struct lattice_struct *lattice)
//##############################################################################
//
// D U M P   P A R A M S 
//
//  - Output the problem parameters to a file.
//
void dump_params( mystruct *param)
{
  FILE *o;
  char filename[1024];

  sprintf( filename,  "./out/params%dx%d.dat", param->LX, param->LY);

  if( !( o = fopen(filename,"w+")))
  {
    printf("%s %d >> ERROR: fopen(\"%s\",\"w+\") = NULL.  Bye, bye!\n", 
        __FILE__, __LINE__, filename);
    exit(1);
  }

  fprintf( o, "LX                   %d\n", param->LX             );
  fprintf( o, "LY                   %d\n", param->LY             );
  fprintf( o, "num_frame            %d\n", param->num_frame );
  fprintf( o, "frame_rate           %d\n", param->frame_rate );
  fprintf( o, "tau0                 %f\n", param->tau0 );
  fprintf( o, "tau1_xx              %f\n", param->tau1_xx );
  fprintf( o, "tau1_yy              %f\n", param->tau1_yy );
  fprintf( o, "gr_x                 %e\n", param->gr_x          );
  fprintf( o, "gr_y                 %e\n", param->gr_y          );
  fprintf( o, "gr_b                 %e\n", param->gr_b          );
  fprintf( o, "ux_in                %lf\n", param->ux_in          );
  fprintf( o, "uy_in                %lf\n", param->uy_in          );
  fprintf( o, "uz_in                %lf\n", param->uz_in          );
  fprintf( o, "rho0_in              %lf\n", param->rho0_in          );
  fprintf( o, "rho0_out             %lf\n", param->rho0_out          );
  fprintf( o, "rho1_in              %lf\n", param->rho1_in          );
  fprintf( o, "rho1_bcs             %lf\n", param->rho1_in          );
  fprintf( o, "sigma_start          %d\n", param->sigma_start          );
  fprintf( o, "btc_pts              %d\n", param->btc_pts          );
  fprintf( o, "btc_spot             %d\n", param->btc_spot          );
  fprintf( o, "dRho                 %lf\n", param->dRho         );
  fprintf( o, "radii                %d\n", param->radii          );
  fprintf( o, "sigma                %lf\n", param->sigma          );
  fprintf( o, "alpha                %lf\n", param->alpha           );
  fprintf( o, "pressure_bcs_ew      %d\n",param->pressure_bcs_ew   );
  fprintf( o, "pressure_bcs_ns      %d\n",param->pressure_bcs_ns   );
  fprintf( o, "solute_bcs_e         %d\n",param->solute_bcs_e   );
  fprintf( o, "solute_bcs_w         %d\n",param->solute_bcs_w   );
  fprintf( o, "solute_bcs_zerograd_n %d\n",param->solute_bcs_zerograd_n   );
  fprintf( o, "solute_bcs_zerograd_s %d\n",param->solute_bcs_zerograd_s   );
  fprintf( o, "solute_bcs_zerograd_e %d\n",param->solute_bcs_zerograd_e   );
  fprintf( o, "solute_bcs_zerograd_w %d\n",param->solute_bcs_zerograd_w   );
  fprintf( o, "solute_bcs_n         %d\n",param->solute_bcs_n   );
  fprintf( o, "solute_bcs_s         %d\n",param->solute_bcs_s   );
  fprintf( o, "restart_frame_num    %d\n",param->restart_frame_num   );

  fclose(o);
  param->N = param->LX*param->LY;
#if TRANSPORT
param->comp=2;
#else
param->comp=1;
#endif

} /* void dump_params( struct lattice_struct *lattice) */
