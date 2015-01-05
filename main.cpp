#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include <stdio.h>
#include <stdlib.h>

#include "include/tiffImageIO.h"
#include "include/tifPreProc.h"

int main(int argc, char *argv[])
{
	char templateFileName[] = "../../../Tiff/4.tif";
	char srcFileName[]      = "../../../Tiff/11.tif";
	char outputFileName[]   = "../../../Tiff/GDALoutput.tif";

	float *tifPixelMatrix = NULL;
	int tifWidth, tifLength;

	showGeoTiffInfo(templateFileName);
	showGeoTiffInfo(srcFileName);

	readTiffImageToMatrix(srcFileName, 1, &tifPixelMatrix, &tifWidth, &tifLength);

    printf("%X \n", tifPixelMatrix[tifWidth * tifLength - 1]);

	tifAdjust(templateFileName, srcFileName, outputFileName);

	/*writeTiffImageRefSrc(outputFileName, srcFileName, 1, tifPixelMatrix);*/

	//GDALDatasetH *templateDataset, *srcDataset;

	//GDALAllRegister();

	//templateDataset = (GDALDatasetH *)GDALOpen(templateFileName, GA_ReadOnly);
	//srcDataset      = (GDALDatasetH *)GDALOpen(srcFileName, GA_ReadOnly);
	//tifAdjust(templateDataset, srcDataset);
	//GDALClose(templateDataset);
	//GDALClose(srcDataset);

	system("PAUSE");

	return 0;
}