/*
 *  GDAL API Tutorial:
 *  http://www.gdal.org/gdal_tutorial.html
 *	gdal.h File Reference
 * 	http://www.gdal.org/gdal_8h.html#a2a74e5e34528589303c1521ebfb9c162
 */

#ifndef _TIFPREPROC_H_
#define _TIFPREPROC_H_

/*
 * Adjust a GeoTiff to a template GeoTiff image and yield an output tiff file.
 */
extern void tifAdjust(GDALDatasetH *templTifDataSet, 
	                  GDALDatasetH *srcTifDataSet, 
	                  GDALDatasetH *outputTifDataSet);

#endif