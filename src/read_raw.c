#define NCHAR 132

void read_raw(int *is_solid, mystruct *param )
{
int LX, LY, NumNodes, i;
FILE  *fp;
double *ns;
char filename[1024];

LX=param->LX; LY=param->LY;NumNodes = param->N;
#if 0 //THREED 
//**************************************************************************
int size_read, size; 
unsigned char *raw;
size = NumNodes*sizeof(unsigned char);
sprintf(filename , "./in/%dx%dx%d.raw", LX,LY,LZ);
fp = fopen( filename, "r+");
if( !fp)
{
printf("%s %d >> ERROR: Can't open file \"%s\" (Exiting!)\n",__FILE__,__LINE__, filename);
exit(1);
}
if( !( raw = (unsigned char *)malloc(size)))
{
printf("%s %d >> read_solids() -- "
"ERROR: Can't malloc image buffer. (Exiting!)\n", __FILE__, __LINE__);
}
printf("%s %d >> Reading %d bytes of %d nodes from file \"%s\".\n",__FILE__, __LINE__, size, NumNodes, filename);
size_read = fread( raw, 1, size, fp);
fclose(fp);
  if( size_read != size)
  {
    printf("%s %d  >> read_solids() -- "
        "ERROR: Can't read image data: read = %d. (Exiting!)\n",
          __FILE__, __LINE__,  size_read);
    exit(1);
  }

for(i=0; i<NumNodes; i++)
is_solid[i] = raw[i]/255;
free(raw);

//**************************************************************************
#else

sprintf(filename , "./in/ns%dx%d.dat", LX,LY);
fp = fopen( filename, "r+");
if( !fp)
{
printf("%s %d >> ERROR: Can't open file \"%s\" (Exiting!)\n",__FILE__,__LINE__, filename);
printf("ns file missing\n SIMULATING A PARALLEL CHANNEL with FLOW in X-DIR----\n");
  	for (i=0; i<param->N; i++) 
	is_solid[i]=(i<param->LX || i >param->N-param->LX || i%param->LX==0 || (i+1)%param->LX==0 )?1:0;
	//is_solid[i]=(i<param->LX || i >param->N-param->LX )?1:0;
	//is_solid[768] =1;
}
else
{

if( !( ns = (double *)malloc(NumNodes*sizeof(double))))
{
printf("%s %d >> read_solids() -- "
"ERROR: Can't malloc image buffer. (Exiting!)\n", __FILE__, __LINE__);
}
for(i=0; i<NumNodes; i++)
{
fscanf(fp, "%lf", &ns[i]);
is_solid[i] = (ns[i]>0)? 1: 0;
}

free(ns);
}
#endif

time_t result;
    result = time(NULL);
    printf("%s \n", asctime(localtime(&result)));
/*
sprintf(filename, "./in/%dx%dx%d.dat",LX,LY,LZ );
fp = fopen( filename, "w");
for(i=0; i<NumNodes; i++)
fprintf(fp,"%d\n", is_solid[i] );
fclose(fp);
*/
return ;
}
