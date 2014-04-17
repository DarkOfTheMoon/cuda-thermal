/* 	This is a Unix port of the bitmap.c code that writes .bmp files to disk. 
	It also runs on Win32, and should be easy to get to run on other platforms. 
	Please visit my web page, http://www.ece.gatech.edu/~slabaugh and click on
	"c" and "Writing Windows Bitmaps" for a further explanation.  This code
	has been tested and works on HP-UX 11.00 using the cc compiler.  To compile,
	just type "cc -Ae bitmapUnix.c" at the command prompt.
	
	If your system is big endian (HP-UX 11.00 is big endian), set the define 
	below to 1.  If it is little endian, set it to 0.  The Windows .bmp format
	is little endian, so if you're running this code on a big endian system
	it will be necessary to swap bytes to write out a little endian file.
	
	Thanks to Robin Pitrat for testing on the Linux platform.

	Greg Slabaugh, 11/05/01
*/	
#define BMP_BIG_ENDIAN 0


#include <stdio.h>
//#include <stdlib.h>
#include <memory.h>

/* This pragma is necessary so that the data in the structures is aligned to 2-byte 
   boundaries.  Some different compilers have a different syntax for this line.  For
   example, if you're using cc on Solaris, the line should be #pragma pack(2).  
*/
#pragma pack (2)


/* Default data types.  Here, uint16 is an unsigned integer that has size 2 bytes (16 bits), 
   and uint32 is datatype that has size 4 bytes (32 bits).  You may need to change these 
   depending on your compiler. */
#define uint16 unsigned short
#define uint32 unsigned int

#define BI_RGB 0
#define BM 19778
#define BMP_FALSE 0
#define BMP_TRUE 1

typedef struct {
   uint16 bfType; 
   uint32 bfSize; 
   uint16 bfReserved1; 
   uint16 bfReserved2; 
   uint32 bfOffBits; 
} BITMAPFILEHEADER; 

typedef struct { 
   uint32 biSize;
   uint32 biWidth; 
   uint32 biHeight; 
   uint16 biPlanes; 
   uint16 biBitCount; 
   uint32 biCompression; 
   uint32 biSizeImage; 
   uint32 biXPelsPerMeter; 
   uint32 biYPelsPerMeter; 
   uint32 biClrUsed; 
   uint32 biClrImportant; 
} BITMAPINFOHEADER; 


typedef struct {
   unsigned char rgbBlue;
   unsigned char rgbGreen;
   unsigned char rgbRed;
   unsigned char rgbReserved;
} RGBQUAD;


/* This function is for byte swapping on big endian systems */
uint16 setUint16(uint16 x)
{
	if (BMP_BIG_ENDIAN)
		return (x & 0x00FF) << 8 | (x & 0xFF00) >> 8;
	else 
		return x;
}

/* This function is for byte swapping on big endian systems */
uint32 setUint32(uint32 x)
{
	if (BMP_BIG_ENDIAN)
		return (x & 0x000000FF) << 24 | (x & 0x0000FF00) << 8 | (x & 0x00FF0000) >> 8 | (x & 0xFF000000) >> 24;
	else 
		return x;
}

/* 
	This function writes out an 8-bit Windows bitmap file that is readable by Microsoft Paint.  
	The image has an arbitrary palette, consisting of up to 256 unique colors.  The image data
        consists of values that index into the palette.
  
   The input to the function is:
           char *filename:				                                A string representing the filename that will be written
                uint32 width:				                            The width, in pixels, of the bitmap
                uint32 height:					                        The height, in pixels, of the bitmap
                unsigned char *image:                                   The image data, where the value indicates an index into the palette
                uint32 numPaletteEntries								The number of entries used in the palette
                RGBQUAD *palette:										The palette

   Written by Greg Slabaugh (slabaugh@ece.gatech.edu), 10/19/00
*/
uint32 write8BitBmpFile(char *filename, uint32 width, uint32 height, unsigned char *image, uint32 numPaletteEntries, RGBQUAD *palette)
{
	BITMAPINFOHEADER bmpInfoHeader;
	BITMAPFILEHEADER bmpFileHeader;
	FILE *filep;
	uint32 row;
	uint32 extrabytes, bytesize;
	unsigned char *paddedImage = NULL;

	/* The .bmp format requires that the image data is aligned on a 4 byte boundary.  For 8 - bit bitmaps,
	   this means that the width of the bitmap must be a multiple of 4. This code determines
	   the extra padding needed to meet this requirement. */
	extrabytes = (4 - width % 4) % 4;

	/* This is the size of the padded bitmap */
	bytesize = (width + extrabytes) * height;

	/* Fill the bitmap file header structure */
	bmpFileHeader.bfType = setUint16(BM);   /* Bitmap header */
	bmpFileHeader.bfSize = setUint32(0);      /* This can be 0 for BI_RGB bitmaps */
	bmpFileHeader.bfReserved1 = setUint16(0);
	bmpFileHeader.bfReserved2 = setUint16(0);
	bmpFileHeader.bfOffBits = setUint32(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * numPaletteEntries);

	/* Fill the bitmap info structure */
	bmpInfoHeader.biSize = setUint32(sizeof(BITMAPINFOHEADER));
	bmpInfoHeader.biWidth = setUint32(width);
	bmpInfoHeader.biHeight = setUint32(height);
	bmpInfoHeader.biPlanes = setUint16(1);
	bmpInfoHeader.biBitCount = setUint16(8);            /* 8 - bit bitmap */
	bmpInfoHeader.biCompression = setUint32(BI_RGB);
	bmpInfoHeader.biSizeImage = setUint32(bytesize);     /* includes padding for 4 byte alignment */
	bmpInfoHeader.biXPelsPerMeter = setUint32(0);
	bmpInfoHeader.biYPelsPerMeter = setUint32(0);
	bmpInfoHeader.biClrUsed = setUint32(numPaletteEntries);
	bmpInfoHeader.biClrImportant = setUint32(0);


	/* Open file */
	if ((filep = fopen(filename, "wb")) == NULL) {
		printf("Error opening file %s\n", filename);
		return BMP_FALSE;
	}

	/* Write bmp file header */
	if (fwrite(&bmpFileHeader, 1, sizeof(BITMAPFILEHEADER), filep) < sizeof(BITMAPFILEHEADER)) {
		printf("Error writing bitmap file header\n");
		fclose(filep);
		return BMP_FALSE;
	}

	/* Write bmp info header */
	if (fwrite(&bmpInfoHeader, 1, sizeof(BITMAPINFOHEADER), filep) < sizeof(BITMAPINFOHEADER)) {
		printf("Error writing bitmap info header\n");
		fclose(filep);
		return BMP_FALSE;
	}

	/* Write bmp palette */
	if (fwrite(palette, 1, numPaletteEntries * sizeof(RGBQUAD), filep) < numPaletteEntries * sizeof(RGBQUAD)) {
		printf("Error writing bitmap palette\n");
		fclose(filep);
		return BMP_FALSE;
	}

	/* Allocate memory for some temporary storage */
	paddedImage = (unsigned char *)calloc(sizeof(unsigned char), bytesize);
	if (paddedImage == NULL) {
		printf("Error allocating memory \n");
		fclose(filep);
		return BMP_FALSE;
	}

	/* Flip image - bmp format is upside down.  Also pad the paddedImage array so that the number
	   of pixels is aligned on a 4 byte boundary. */
	for (row = 0; row < height; row++)
		memcpy(&paddedImage[row * (width + extrabytes)], &image[(height - 1 - row) * width], width);
	
	/* Write bmp data */
	if (fwrite(paddedImage, 1, bytesize, filep) < bytesize) {
		printf("Error writing bitmap data\n");
		free(paddedImage);
		fclose(filep);
		return BMP_FALSE;
	}

	/* Close file */
	fclose(filep);
	free(paddedImage);
	return BMP_TRUE;
}

/* 
	This function writes out a 24-bit Windows bitmap file that is readable by Microsoft Paint.  
	The image data is a 1D array of (r, g, b) triples, where individual (r, g, b) values can 
	each take on values between 0 and 255, inclusive.
  
   The input to the function is:
	char *filename:					A string representing the filename that will be written
	uint32 width:					The width, in pixels, of the bitmap
	uint32 height:					The height, in pixels, of the bitmap
	unsigned char *image:				The image data, where each pixel is 3 unsigned chars (r, g, b)

   Written by Greg Slabaugh (slabaugh@ece.gatech.edu), 10/19/00
*/
uint32 write24BitBmpFile(char *filename, uint32 width, uint32 height, unsigned char *image)
{
	BITMAPINFOHEADER bmpInfoHeader;
	BITMAPFILEHEADER bmpFileHeader;
	FILE *filep;
	uint32 row, column;
	uint32 extrabytes, bytesize;
	unsigned char *paddedImage = NULL, *paddedImagePtr, *imagePtr;

	extrabytes = (4 - (width * 3) % 4) % 4;

        /* This is the size of the padded bitmap */
        bytesize = (width * 3 + extrabytes) * height;

        /* Fill the bitmap file header structure */
        bmpFileHeader.bfType = setUint16(BM);   /* Bitmap header */
        bmpFileHeader.bfSize = setUint32(0);      /* This can be 0 for BI_RGB bitmaps */
        bmpFileHeader.bfReserved1 = setUint16(0);
        bmpFileHeader.bfReserved2 = setUint16(0);
        bmpFileHeader.bfOffBits = setUint32(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER));

        /* Fill the bitmap info structure */
        bmpInfoHeader.biSize = setUint32(sizeof(BITMAPINFOHEADER));
        bmpInfoHeader.biWidth = setUint32(width);
        bmpInfoHeader.biHeight = setUint32(height);
        bmpInfoHeader.biPlanes = setUint16(1);
        bmpInfoHeader.biBitCount = setUint16(24);            /* 24 - bit bitmap */
        bmpInfoHeader.biCompression = setUint32(BI_RGB);
        bmpInfoHeader.biSizeImage = setUint32(bytesize);     /* includes padding for 4 byte alignment */
        bmpInfoHeader.biXPelsPerMeter = setUint32(0);
        bmpInfoHeader.biYPelsPerMeter = setUint32(0);
        bmpInfoHeader.biClrUsed = setUint32(0);
        bmpInfoHeader.biClrImportant = setUint32(0);


        /* Open file */
        if ((filep = fopen(filename, "wb")) == NULL) {
                printf("Error opening file %s\n", filename);
                return BMP_FALSE;
        }

        /* Write bmp file header */
        if (fwrite(&bmpFileHeader, 1, sizeof(BITMAPFILEHEADER), filep) < sizeof(BITMAPFILEHEADER)) {
                printf("Error writing bitmap file header\n");
                fclose(filep);
                return BMP_FALSE;
        }

        /* Write bmp info header */
        if (fwrite(&bmpInfoHeader, 1, sizeof(BITMAPINFOHEADER), filep) < sizeof(BITMAPINFOHEADER)) {
                printf("Error writing bitmap info header\n");
                fclose(filep);
                return BMP_FALSE;
        }
     
        
	/* Allocate memory for some temporary storage */
	paddedImage = (unsigned char *)calloc(sizeof(unsigned char), bytesize);
	if (paddedImage == NULL) {
		printf("Error allocating memory \n");
		fclose(filep);
		return BMP_FALSE;
	}

	/* This code does three things.  First, it flips the image data upside down, as the .bmp
	format requires an upside down image.  Second, it pads the image data with extrabytes 
	number of bytes so that the width in bytes of the image data that is written to the
	file is a multiple of 4.  Finally, it swaps (r, g, b) for (b, g, r).  This is another
	quirk of the .bmp file format. */
	
	for (row = 0; row < height; row++) {
		imagePtr = image + (height - 1 - row) * width * 3;
		paddedImagePtr = paddedImage + row * (width * 3 + extrabytes);
		for (column = 0; column < width; column++) {
			*paddedImagePtr = *(imagePtr + 2);
			*(paddedImagePtr + 1) = *(imagePtr + 1);
			*(paddedImagePtr + 2) = *imagePtr;
			imagePtr += 3;
			paddedImagePtr += 3;
		}
	}

	/* Write bmp data */
	if (fwrite(paddedImage, 1, bytesize, filep) < bytesize) {
		printf("Error writing bitmap data\n");
		free(paddedImage);
		fclose(filep);
		return BMP_FALSE;
	}

	/* Close file */
	fclose(filep);
	free(paddedImage);
	return BMP_TRUE;
}


/* 
	This function writes out a grayscale image as a 8-bit Windows bitmap file that is readable by Microsoft Paint.
	It creates a palette and then calls write8BitBmpFile to output the bitmap file.
  
   The input to the function is:
	   char *filename:						A string representing the filename that will be written
		uint32 width:					The width, in pixels, of the bitmap
		uint32 height:					The height, in pixels, of the bitmap
		unsigned char *image:				The image data, where the value indicates a color between 0 (black) and 255 (white)

   Written by Greg Slabaugh (slabaugh@ece.gatech.edu), 10/19/00
*/
uint32 writeGrayScaleDataToBmpFile(char *filename, uint32 width, uint32 height, unsigned char *image)
{
	RGBQUAD palette[256];
	uint32 i;
	uint32 numPaletteEntries = 256;

	/* Create the palette - each pixel is an index into the palette */
	for (i = 0; i < numPaletteEntries; i++) {
		palette[i].rgbRed = i;
		palette[i].rgbGreen = i;
		palette[i].rgbBlue = i;
		palette[i].rgbReserved = 0;
	}

   return write8BitBmpFile(filename, width, height, image, numPaletteEntries, palette);
	
}
//
//
int LoadShadabsData(char *filename_fl_dat, char *filename_sol_dat,
                    int width, int height, unsigned char *imageData8Bit, 
                    unsigned char *imageData8Bit_sol, int dynamic_scale, int comp,
                    int *is_solid, double *rho)
{
	int i; 
	float min = 1000000;
	float max = 0;
	// Get max and min
	for (i = 0; i < width * height; i++) {
		max = (rho[i] > max) ? rho[i] : max;
		min = (rho[i] < min) ? rho[i] : min ;

if(comp==1)
    min = (min<1e-12) ? 0. : min;

	}

	// Convert to unsigned char
	for (i = 0; i < width * height; i++) 
  {
    if(dynamic_scale)
    {
		imageData8Bit[i] = (unsigned char) (255 * (rho[i] - min) / (max - min));
    if( (max-min)<1e-09)
		imageData8Bit[i] = (unsigned char) (255);

    }
    else
    {

		imageData8Bit[i] = (unsigned char)  (255*rho[i] ) ;

    }

  }

//	free(floatImage);
	
	return 1;

}

#define DEBUG 2
int bitmap(int lx, int ly, int frame, int color_scale, int comp, int *is_solid, double *rho )
{

	int x, y, index ;
	unsigned char *imageData8Bit = NULL;
	unsigned char *imageData24Bit = NULL;
  char filename_fl_dat[1024], filename_fl_bmp[1024];

	unsigned char *imageData8Bit_sol  = NULL;
	unsigned char *imageData24Bit_sol = NULL;
  char filename_sol_dat[1024], filename_sol_bmp[1024];
  //int  frame;
	RGBQUAD palette[4];
	uint32 i;

	if (DEBUG == 0) {
		printf("Testing grayscale code\n");
		for (i = 0; i < lx * ly; i++)
			imageData8Bit[i] = 0;

		imageData8Bit[0] = 255;
		imageData8Bit[1] = 100;
		imageData8Bit[lx] = 100;
		imageData8Bit[lx * ly - 1] = 255;
	   	writeGrayScaleDataToBmpFile("test0.bmp", lx, ly, imageData8Bit);
	   
	} else if (DEBUG == 1) {
		printf("Testing 8-bit code\n");

		palette[0].rgbRed = 0;
		palette[0].rgbGreen = 0;
		palette[0].rgbBlue = 0;
		palette[0].rgbReserved = 0;

		palette[1].rgbRed = 255;
		palette[1].rgbGreen = 0;
		palette[1].rgbBlue = 0;
		palette[1].rgbReserved = 0;

		palette[2].rgbRed = 0;
		palette[2].rgbGreen = 255;
		palette[2].rgbBlue = 0;
		palette[2].rgbReserved = 0;

		palette[3].rgbRed = 0;
		palette[3].rgbGreen = 0;
		palette[3].rgbBlue = 255;
		palette[3].rgbReserved = 0;

		for (i = 0; i < lx * ly; i++)
		imageData8Bit[i] = 0;

		imageData8Bit[0] = 1;
		imageData8Bit[1] = 2;
		imageData8Bit[lx] = 3;
		imageData8Bit[lx * ly - 1] = 1;

		write8BitBmpFile("test1.bmp", lx, ly, imageData8Bit, 4, palette);

	} else {
	//	printf("Testing 24-bit code\n");

	imageData8Bit = (unsigned char *) malloc(lx * ly * sizeof(unsigned char));
	if (imageData8Bit == NULL)
		return 0;

	imageData24Bit = (unsigned char *) malloc(3 * lx * ly * sizeof(unsigned char));
	if (imageData24Bit == NULL)
		return 0;

	imageData8Bit_sol = (unsigned char *) malloc(lx * ly * sizeof(unsigned char));
	if (imageData8Bit_sol == NULL)
		return 0;

	imageData24Bit_sol = (unsigned char *) malloc(3 * lx * ly * sizeof(unsigned char));
	if (imageData24Bit_sol == NULL)
		return 0;

      //for(frame=0; frame<=numframe; frame++)
      {
    sprintf( filename_fl_dat, "./out/rho[0]%dx%d_t%03d.dat", 
     lx,ly,frame);
if(comp==0)
    sprintf( filename_fl_bmp, "./out/rho[0]%dx%d_frame%03d.bmp", 
     lx,ly,frame);
else
    sprintf( filename_fl_bmp, "./out/rho[1]%dx%d_frame%03d.bmp", 
     lx,ly,frame);
    sprintf( filename_sol_dat, "./out/rho[1]%dx%d_t%03d.dat", 
     lx,ly,frame);
    sprintf( filename_sol_bmp, "./out/rho[1]_%dx%d_frame%03d.bmp", 
     lx,ly,frame);
    //printf("frame=%d\n",frame);
	LoadShadabsData(filename_fl_dat, filename_sol_dat, lx, ly, 
                  imageData8Bit, imageData8Bit_sol, color_scale, comp, is_solid, rho );

	// Put data into different color channels
	for (y = 0; y < ly; y++) {
		for (x = 0; x < lx; x++) {
			index = x + y * lx;

    
#if FREED_PM

	if(comp==0)
	{
			imageData24Bit[3*index+2] =   255;//BLUE
			imageData24Bit[3*index+0] = 255-imageData8Bit[index]; //BLUE
			imageData24Bit[3*index+1] = 255-imageData8Bit[index]; //BLUE
	}
	else
	{

			imageData24Bit[3*index+0] =   255;//RED
			imageData24Bit[3*index+2] = 255-imageData8Bit[index]; //RED
			imageData24Bit[3*index+1] = 255-imageData8Bit[index]; //RED
	}
#else

      if(is_solid[index]==1)
    	{
    //printf(" is_solid=%d\n", is_solid[x][y]);
			imageData24Bit[3*index+2] = 0;
			imageData24Bit[3*index+0] = 0;
			imageData24Bit[3*index+1] = 0;

    	}
      else
      {
	if(comp==0)
	{
			imageData24Bit[3*index+2] =   255;//BLUE
			imageData24Bit[3*index+0] = 255-imageData8Bit[index]; //BLUE
			imageData24Bit[3*index+1] = 255-imageData8Bit[index]; //BLUE
	}
	else
	{

			imageData24Bit[3*index+0] =   255;//RED
			imageData24Bit[3*index+2] = 255-imageData8Bit[index]; //RED
			imageData24Bit[3*index+1] = 255-imageData8Bit[index]; //RED
	}
      }
#endif

		}
	}

	write24BitBmpFile(filename_fl_bmp,  lx, ly, imageData24Bit);
	//write24BitBmpFile(filename_sol_bmp, lx, ly, imageData24Bit_sol);
	 

      }

	}

	free(imageData8Bit);
	free(imageData24Bit);

	return BMP_TRUE;
}
