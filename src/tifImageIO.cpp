/*
 *  GDAL installation and configuration in Visual Studio 2010 references:
 *  http://www.cnblogs.com/bigbigtree/archive/2011/11/19/2255495.html
 *  http://xzh2012.blog.163.com/blog/static/114980038201332993145885/
 *
 *  GDAL API Tutorial:
 *  http://www.gdal.org/gdal_tutorial.html
 *  gdal.h File Reference
 *  http://www.gdal.org/gdal_8h.html#a2a74e5e34528589303c1521ebfb9c162
 */

#include <stdio.h>  
#include <stdlib.h>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include "../include/tiffImageIO.h"

/*
 * Read pixel value of tiff file into an array.
 */
void readTiffImageToMatrix(const char srcFileName[], int bandId, float **tifPixelMatrix, 
                           int *tifWidth, int *tifLength) 
{
    GDALDatasetH *poDataset;

    GDALAllRegister();

    poDataset = (GDALDatasetH *)GDALOpen(srcFileName, GA_ReadOnly);

    if ( poDataset != NULL )
    {
        GDALRasterBandH hBand;
        hBand = GDALGetRasterBand( poDataset, bandId );

        int   nXSize = GDALGetRasterXSize(poDataset);
        int   nYSize = GDALGetRasterYSize(poDataset);
        *tifWidth = nXSize;
        *tifLength = nYSize;

        *tifPixelMatrix = (float *) CPLMalloc(sizeof(float) * nXSize * nYSize);
        GDALRasterIO( hBand, GF_Read, 0, 0, nXSize, nYSize, 
                      *tifPixelMatrix, nXSize, nYSize, GDT_Float32, 
                      0, 0 );

        GDALClose(poDataset);
    }
}

/*
 * Read pixel value of tiff data-set into an array.
 */
void readTifDataSetToMatrix(GDALDatasetH *srcTifDataSet, int bandId, float **tifPixelMatrix)
{
    if ( srcTifDataSet != NULL )
    {
        GDALRasterBandH hBand;
        hBand = GDALGetRasterBand( srcTifDataSet, bandId );

        int   nXSize = GDALGetRasterXSize(srcTifDataSet);
        int   nYSize = GDALGetRasterYSize(srcTifDataSet);

        *tifPixelMatrix = (float *) CPLMalloc(sizeof(float) * nXSize * nYSize);
        GDALRasterIO( hBand, GF_Read, 0, 0, nXSize, nYSize, 
                      *tifPixelMatrix, nXSize, nYSize, GDT_Float32, 
                      0, 0 );
    }
}

/*
 * Create a tif file using the config info of a source tif file 
 * and write pixel value matrix to it.
 */
void writeTiffImageRefSrc(const char dstFileName[], const char srcFileName[], 
                          int bandId, float *pixelMatrixBuf) 
{
        const char *pszFormat = "GTiff";
        GDALDriverH hDriver = GDALGetDriverByName( pszFormat );

        GDALDatasetH hSrcDS = GDALOpen( srcFileName, GA_ReadOnly );

        if ( hSrcDS != NULL ) {
            GDALDatasetH hDstDS = GDALCreateCopy( hDriver, dstFileName, hSrcDS, FALSE, 
                NULL, NULL, NULL );

            GDALClose(hSrcDS);

            if( hDstDS == NULL ) {
                GDALClose( hDstDS );
                printf("Failed creating output file!\n");
                return;
            }

            int   nXSize = GDALGetRasterXSize(hDstDS);
            int   nYSize = GDALGetRasterYSize(hDstDS);

            GDALRasterBandH hBand;
            hBand = GDALGetRasterBand( hDstDS, bandId );

            GDALRasterIO( hBand, GF_Write, 0, 0, nXSize, nYSize, 
                          pixelMatrixBuf, nXSize, nYSize, GDT_Float32, 
                          0, 0 );

            GDALClose(hDstDS);
        }
}

/*
 * Print GeoTiff info.
 */
void showGeoTiffInfo(char srcFileName[]) 
{
    GDALDatasetH *poDataset;
    double        adfGeoTransform[6];

    GDALAllRegister();

    poDataset = (GDALDatasetH *)GDALOpen(srcFileName, GA_ReadOnly);

    if ( poDataset != NULL )
    {
        printf("========================GeoTiff Info========================= \n");
        printf("Tiff File: %s\n", srcFileName);

        printf("RasterXSize: %d\n", GDALGetRasterXSize(poDataset));

        printf("RasterYSize: %d\n", GDALGetRasterYSize(poDataset));

        printf("RasterCount: %d\n",  GDALGetRasterCount(poDataset));

        if( GDALGetProjectionRef( poDataset ) != NULL )
            printf( "Projection is `%s'\n", GDALGetProjectionRef( poDataset ) );

        if( GDALGetGeoTransform( poDataset, adfGeoTransform ) == CE_None )
        {
            printf( "Origin = (%.6f,%.6f)\n",
                adfGeoTransform[0], adfGeoTransform[3] );

            printf( "Pixel Size = (%.6f,%.6f)\n",
                adfGeoTransform[1], adfGeoTransform[5] );
        }

        double rasterMinMax[2];
        GDALRasterBandH hBand;
        for ( int i = 1; i <= GDALGetRasterCount(poDataset); i++ )
        {
            hBand = GDALGetRasterBand( poDataset, i );
            GDALComputeRasterMinMax( hBand, 1, rasterMinMax );
            printf( "Raster band %d pixel value ranges [%f, %f] \n", i, rasterMinMax[0], rasterMinMax[1] );
        }

        printf("==============================End============================ \n");

        GDALClose(poDataset);
    }
}