/*
 *  GDAL API Tutorial:
 *  http://www.gdal.org/gdal_tutorial.html
 *	gdal.h File Reference
 * 	http://www.gdal.org/gdal_8h.html#a2a74e5e34528589303c1521ebfb9c162
 */

#ifndef _TIFFIMAGEIO_H_
#define _TIFFIMAGEIO_H_

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

/*
 * Read pixel value of tiff file into an array.
 */
extern void readTiffImageToMatrix( const char srcFileName[], 
	                               int        bandId, 
						           float      **tifPixelMatrix, 
	                               int        *tifWidth, 
								   int        *tifLength,
						           double     *rasterMinMax ); 

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