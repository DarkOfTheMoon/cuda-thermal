#define BL   256

# define Q 9
# define DIM 2
# define IC 0
# define PARABOLIC_ICS 0

#include "params.h"
#include "read_params.h"
#include "init_f_mrt.cu"
#include "init_vars.c"
#include "macro_vars.cu"
#include "write_data.c"
#include "collision.cu"
#include "stream.cu"
#include "bitmapUnix.c"
#include "read_raw.c"
#include "bcs_solute.cu"
#include "bcs_fluid.cu"
