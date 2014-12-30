#include <stdio.h>
#include <stdlib.h>

#include <GDAL/gdal.h>
#include <GDAL/gdal_priv.h>
#include <GDAL/cpl_conv.h>

/*
 *
 */
void tifAdjust(GDALDatasetH *templateTifDataSet,
               GDALDatasetH *srcTifDataSet,
               GDALDatasetH *outputTifDataSet)
{

}


/*
 *
 */
bool Projection2ImageRowCol(double *adfGeoTransform, double dProjX, double dProjY,
                            int &iCol, int &iRow)
{
    try
    {
        double dTemp = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] * adfGeoTransform[4];
        double dCol = 0.0, dRow = 0.0;
        dCol = (adfGeoTransform[5] * (dProjX - adfGeoTransform[0]) -
                adfGeoTransform[2] * (dProjY - adfGeoTransform[3])) / dTemp + 0.5;
        dRow = (adfGeoTransform[1] * (dProjY - adfGeoTransform[3]) -
                adfGeoTransform[4] * (dProjX - adfGeoTransform[0])) / dTemp + 0.5;

        iCol = static_cast<int>(dCol);
        iRow = static_cast<int>(dRow);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

/*
 *
 */
bool ImageRowCol2Projection(double *adfGeoTransform, int iCol, int iRow,
                            double &dProjX, double &dProjY)
{
    //adfGeoTransform[6]  数组adfGeoTransform保存的是仿射变换中的一些参数，分别含义见下
    //adfGeoTransform[0]  左上角x坐标
    //adfGeoTransform[1]  东西方向分辨率
    //adfGeoTransform[2]  旋转角度, 0表示图像 "北方朝上"
    //adfGeoTransform[3]  左上角y坐标
    //adfGeoTransform[4]  旋转角度, 0表示图像 "北方朝上"
    //adfGeoTransform[5]  南北方向分辨率

    try
    {
        dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow;
        dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow;
        return true;
    }
    catch (...)
    {
        return false;
    }
}