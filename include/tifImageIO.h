/*
 *  GDAL API Tutorial:
 *  http://www.gdal.org/gdal_tutorial.html
 *	gdal.h File Reference
 * 	http://www.gdal.org/gdal_8h.html#a2a74e5e34528589303c1521ebfb9c162
 */

#ifndef _TIFFIMAGEIO_H_
#define _TIFFIMAGEIO_H_

/*
 * Read pixel value of tiff file into an array.
 */
extern void readTiffImageToMatrix(const char srcFileName[], int bandId, float **tiffMatrix, 
	                              int *tiffWidth, int *tiffLength);

/*
 * Read pixel value of tiff data-set into an array.
 */
extern void readTifDataSetToMatrix(GDALDatasetH *srcTifDataSet, int bandId, float **tifPixelMatrix);

/*
 * Create a tif file using the config info of a source tif file 
 * and write pixel value matrix to it.
 */
extern void writeTiffImageRefSrc(const char dstFileName[], const char srcFileName[], 
	                             int bandid, float *pixelMatrixBuf);

/*
 * Print GeoTiff info.
 */
extern void showGeoTiffInfo(char srcFileName[]);    

#endif