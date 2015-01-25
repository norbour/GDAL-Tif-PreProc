/*
 *  GDAL API Tutorial:
 *  http://www.gdal.org/gdal_tutorial.html
 *	gdal.h File Reference
 * 	http://www.gdal.org/gdal_8h.html#a2a74e5e34528589303c1521ebfb9c162
 */

#ifndef _TIFFIMAGEIO_H_
#define _TIFFIMAGEIO_H_

 #ifndef HAVE_INT8
typedef	signed char int8;	/* NB: non-ANSI compilers may not grok */
#endif
typedef	unsigned char uint8;
#ifndef HAVE_INT16
typedef	short int16;
#endif
typedef	unsigned short uint16;	/* sizeof (uint16) must == 2 */
#if SIZEOF_INT == 4
#ifndef HAVE_INT32
typedef	int int32;
#endif
typedef	unsigned int uint32;	/* sizeof (uint32) must == 4 */
#elif SIZEOF_LONG == 4
#ifndef HAVE_INT32
typedef	long int32;
#endif
typedef	unsigned long uint32;	/* sizeof (uint32) must == 4 */
#endif

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

/*
 * Get GeoTiff min & max raster pixel value.
 */   
extern void getTiffMinMax( const char srcFileName[],
	                       int        bandId,
				           double     *rasterMinMax,
				           int        bApproxOK );

/*
 * Get GeoTiff raster width & length.
 */
extern void getTiffWidthLength( const char srcFileName[],
	                            int        bandId,
						        int        *tifWidth,
						        int        *tifLength );

/*
 * Read pixel value of tiff file into an array.
 */
extern void readTiffImageToMatrix( const char srcFileName[], 
	                               int        bandId, 
						           float      **tifPixelMatrix ); 

/*
 * Read pixel value of tiff data-set into an array.
 */
extern void readTifDataSetToMatrix( GDALDatasetH *srcTifDataSet, 
                                    int          bandId, 
								    float        **tifPixelMatrix );

/*
 * Create a tif file using the config info of a source tif file 
 * and write pixel value matrix to it.
 */
extern void writeTiffImageRefSrc( const char dstFileName[], 
	                              const char srcFileName[], 
	                              int        bandId, 
								  float      *pixelMatrixBuf );

/*
 * Copy raster statistics values.
 */
extern void copyRasterStatistics( const char dstTifFile[], 
	                              const char srcTifFile[], 
	                              int        dstBandId, 
								  int        srcBandId,
						          int        bApproxOK );

/**
 * Alter raster pixel min & max values.
 */
extern void alterRasterMinMax( const char   tifFile[], 
	                           int          bandId,
						       const double newRasterMinMax[2] );

/*
 * Print GeoTiff info.
 */
extern void showGeoTiffInfo( const char srcFileName[] );    

#endif