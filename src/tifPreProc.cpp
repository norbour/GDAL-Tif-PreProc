#include <stdio.h>  
#include <stdlib.h>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include "../include/tifPreProc.h"

/*
 * Adjust a GeoTiff to a template GeoTiff image and yield an output tiff file.
 */
void tifAdjust(GDALDatasetH *templTifDataSet, 
               GDALDatasetH *srcTifDataSet, 
               GDALDatasetH *outputTifDataSet)
{
    double templTifGeoTransform[6], srcTifGeoTransform[6];
    int srcSubOriginX = 0, srcSubOriginY = 0;

    GDALGetGeoTransform( templTifDataSet, templTifGeoTransform );
    GDALGetGeoTransform( srcTifDataSet,   srcTifGeoTransform );

    Projection2ImageRowCol(srcTifGeoTransform, templTifGeoTransform[0], templTifGeoTransform[3],
                           &srcSubOriginX, &srcSubOriginY);

    printf("Source tif sub-origin coordinates is [%d, %d].\n", srcSubOriginX, srcSubOriginY);
}

/*
 * Translate geographical coordinates to image row&col value according to its geographical config info.
 */
void Projection2ImageRowCol(double *adfGeoTransform, double dProjX, double dProjY,
                            int &iCol, int &iRow)
{
    double dTemp = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] * adfGeoTransform[4];
    double dCol = 0.0, dRow = 0.0;
    dCol = (adfGeoTransform[5] * (dProjX - adfGeoTransform[0]) -
            adfGeoTransform[2] * (dProjY - adfGeoTransform[3])) / dTemp + 0.5;
    dRow = (adfGeoTransform[1] * (dProjY - adfGeoTransform[3]) -
            adfGeoTransform[4] * (dProjX - adfGeoTransform[0])) / dTemp + 0.5;

    iCol = (int)dCol;
    iRow = (int)dRow;
}

/*
 * Translate geographical coordinates to image row&col value according to its geographical config info.
 */
void ImageRowCol2Projection(double *adfGeoTransform, int iCol, int iRow,
                            double &dProjX, double &dProjY)
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
    dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow;
    dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow;
}