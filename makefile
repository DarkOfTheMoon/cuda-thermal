################################################################################
#
#
# Makefile
#
EXE=new.exe
all:
	nvcc -arch sm_13 -o new.exe ./src/mcmp.cu
mod:
	nvcc -arch sm_13 -o new.exe ./src_bgk_mod/mcmp.cu
mrt:
	nvcc -arch sm_13 -o new.exe ./src_mrt/mcmp.cu
sweep:
	/bin/rm -f out/*.dat
	/bin/rm -f out/*.bmp
	/bin/rm -f *.*~
	/bin/rm -f .*.swp
