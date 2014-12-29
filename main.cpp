#include <GDAL/gdal.h>
#include <GDAL/gdal_priv.h>
#include <GDAL/cpl_conv.h>

#include <stdio.h>
#include <stdlib.h>

#include "include/tifImageIO.h"

int main(int argc, char *argv[])
{
	char srcFileName[] = "tif/9.tif";
	char dstFileName[] = "tif/GDALoutput.tif";

	float *tifPixelMatrix = NULL;
	int tifWidth, tifLength;

	showGeoTiffInfo(srcFileName);

	readTiffImageToMatrix(srcFileName, 1, &tifPixelMatrix, &tifWidth, &tifLength);

    printf("%x \n", tifPixelMatrix[tifWidth * tifLength - 1]);

	writeTiffImageRefSrc(dstFileName, srcFileName, 1, tifPixelMatrix);

	system("PAUSE");

	return 0;
}