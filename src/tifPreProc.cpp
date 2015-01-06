#include <stdio.h>  
#include <stdlib.h>
#include <string.h>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include "../include/tifPreProc.h"
#include "../include/tiffImageIO.h"

#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#define MIN(a,b) ( ((a)>(b)) ? (b):(a) )

/*
 * Translate geographical coordinates to image row&col value according to its geographical config info.
 */
void Projection2ImageRowCol(double *adfGeoTransform, double dProjX, double dProjY,
                            int *iCol, int *iRow)
{
    double dTemp = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] * adfGeoTransform[4];
    double dCol = 0.0, dRow = 0.0;
    dCol = (adfGeoTransform[5] * (dProjX - adfGeoTransform[0]) -
            adfGeoTransform[2] * (dProjY - adfGeoTransform[3])) / dTemp + 0.5;
    dRow = (adfGeoTransform[1] * (dProjY - adfGeoTransform[3]) -
            adfGeoTransform[4] * (dProjX - adfGeoTransform[0])) / dTemp + 0.5;

    *iCol = (int)dCol;
    *iRow = (int)dRow;
}

/*
 * Translate geographical coordinates to image row&col value according to its geographical config info.
 */
void ImageRowCol2Projection(double *adfGeoTransform, int iCol, int iRow,
                            double *dProjX, double *dProjY)
{
    /* 
     * adfGeoTransform[6]  Array adfGeoTransform[6] keep affine transformation parameters of a GeoTiff:
     * adfGeoTransform[0]  Top left corner x coordinates value
     * adfGeoTransform[1]  pixel grid cell dx (East-West direction resolution)
     * adfGeoTransform[2]  Rotation angle, 0 means image upper-North
     * adfGeoTransform[3]  Top left corner y coordinates value
     * adfGeoTransform[4]  Rotation angle, 0 means image upper-North
     * adfGeoTransform[5]  pixel grid cell dy (South-North direction resolution)
     */
    *dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow;
    *dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow;
}

/*
 * <Core Function>
 * Adjust a GeoTiff to a template GeoTiff image and yield an output tiff file.
 */
void tifAdjustCore(GDALDatasetH *templateTifDataSet, 
                   GDALDatasetH *srcTifDataSet,
                   float        **adjustedPixelMatrix)
{
    double templTifGeoTransform[6], srcTifGeoTransform[6];
    int srcSubOriginCol = 0, srcSubOriginRow = 0;

    GDALGetGeoTransform( templateTifDataSet, templTifGeoTransform );
    GDALGetGeoTransform( srcTifDataSet,      srcTifGeoTransform );

    Projection2ImageRowCol(srcTifGeoTransform, templTifGeoTransform[0], templTifGeoTransform[3],
        &srcSubOriginCol, &srcSubOriginRow);

    int templateTifWidth  = GDALGetRasterXSize(templateTifDataSet);
    int templateTifHeight = GDALGetRasterYSize(templateTifDataSet);

    *adjustedPixelMatrix = (float*)malloc(sizeof(float) * templateTifWidth * templateTifHeight);

    int srcTifWidth  = GDALGetRasterXSize(srcTifDataSet); 
    int srcTifHeight = GDALGetRasterYSize(srcTifDataSet);
    float *srcPixelMatrix = NULL;

    readTifDataSetToMatrix(srcTifDataSet, 1, &srcPixelMatrix);

    int upLeftX      = MAX(0, srcSubOriginCol);
    int upLeftY      = MAX(0, srcSubOriginRow);
    int buttomRightX = MIN(srcTifWidth - 1, srcSubOriginCol + templateTifWidth - 1);
    int buttomRightY = MIN(srcTifHeight - 1, srcSubOriginRow + templateTifHeight - 1);

    for (int j = srcSubOriginRow; j < templateTifHeight + srcSubOriginRow; j++)
    {
        for (int i = srcSubOriginCol; i < templateTifWidth + srcSubOriginCol; i++) 
        {
            if ( j >= upLeftY && j <= buttomRightY && i >= upLeftX && i <= buttomRightX)
            {
                /**(*adjustedPixelMatrix + (i - srcSubOriginCol + (j - srcSubOriginRow) * templateTifWidth))
                = srcPixelMatrix[i + j * srcTifWidth];*/
                memcpy((*adjustedPixelMatrix + (i - srcSubOriginCol + (j - srcSubOriginRow) * templateTifWidth)),
                       srcPixelMatrix + (i + j * srcTifWidth), sizeof(float));
            }
            else
            {
                /**(*adjustedPixelMatrix + (i - srcSubOriginCol + (j - srcSubOriginRow) * templateTifWidth))
                = 0xE0000000;*/
                // The pixel value of outside-zone place was supposed to be 0xE0000000.
                memcpy((*adjustedPixelMatrix + (i - srcSubOriginCol + (j - srcSubOriginRow) * templateTifWidth)), 
                       srcPixelMatrix, sizeof(float)); 
            }
        }
    }
}

/*
 * <Interface>
 * Adjust a GeoTiff to a template GeoTiff image and yield an output tiff file.
 */
void tifAdjust(const char templateTifFile[], 
               const char srcTifFile[],
               const char outputTifFile[])
{
    GDALDatasetH *templateTifDataSet, *srcTifDataSet;

    GDALAllRegister();

    templateTifDataSet = (GDALDatasetH *)GDALOpen(templateTifFile, GA_ReadOnly);
    srcTifDataSet      = (GDALDatasetH *)GDALOpen(srcTifFile, GA_ReadOnly);

    if (templateTifDataSet != NULL && srcTifDataSet != NULL)
    {
        float *adjustedPixelMatrix = NULL;
        tifAdjustCore(templateTifDataSet, srcTifDataSet, &adjustedPixelMatrix);

        writeTiffImageRefSrc(outputTifFile, templateTifFile, 1, adjustedPixelMatrix);

        GDALClose(templateTifDataSet);
        GDALClose(srcTifDataSet);

        /* CreateCopy method in writeTiffImageRefSrc() copy the statistics value (mainly Min/Max pixel sample value) 
           from template tiff into adjusted tiff, we need to repair this. */
        copyRasterStatistics(outputTifFile, srcTifFile, 1, 1, 1);
    }
}