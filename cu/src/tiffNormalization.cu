#include <stdio.h>
#include <stdlib.h>	
#include <math.h>
 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/book.h"
#include "../include/commCuda.h"
#include "../../include/tiffImageIO.h"

/**
 * @Device
 * Active factor GeoTiff pixel normalization.
 */
__global__ void normalizeActiveRasterPixel( float  *pixelMatrix, 
	                                        int    *nPixels, 
										    double *rasterMinMax ) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < *nPixels)
	{
		
		if ( pixelMatrix[tid] - 0xE0000000 != 0 ) // float nullPixelValue = 0xE0000000;
		{
			pixelMatrix[tid] = (pixelMatrix[tid] - rasterMinMax[0]) /
				               (rasterMinMax[1]  - rasterMinMax[0]);
		}
		
		tid += blockDim.x * gridDim.x; 
	}
}

/**
 * @Device
 * Negative factor GeoTiff pixel normalization.
 */
__global__ void normalizeNegativeRasterPixel( float  *pixelMatrix, 
	                                          int    *nPixels, 
	                                          double *rasterMinMax ) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < *nPixels)
	{
		pixelMatrix[tid] = (rasterMinMax[1] - pixelMatrix[tid]) /
			               (rasterMinMax[1] - rasterMinMax[0]);

		tid += blockDim.x * gridDim.x; 
	}
}

/**
 * <Core Function>
 * Factor GeoTiff pixel normalization.
 * @param pixelMatrix  -> raster pixel value array
 * @param tiffWidth    -> raster width
 * @param tiffHeigth   -> raster length
 * @param rasterMinMax -> min & max value in raster pixels
 * @param factorType   -> evaluation factor type (Active/Negative)
 */
void rasterPixelNormalization(float         *pixelMatrix, 
	                          int           tiffWidth, 
							  int           tiffHeigth, 
					          const double  rasterMinMax[2],
                              envFactorType factorType) 
{
	int nPixels = tiffWidth * tiffHeigth;

	float  *dev_pixelMatrix  = NULL;    
	int    *dev_nPixels      = NULL;    
	double *dev_rasterMinMax = NULL;     

	HANDLE_ERROR( cudaMalloc( (void**)&dev_pixelMatrix, nPixels * sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_nPixels,               sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_rasterMinMax,      2 * sizeof(double) ) );

	HANDLE_ERROR( cudaMemcpy( dev_pixelMatrix,
		                      pixelMatrix,
							  nPixels * sizeof(float),
							  cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_nPixels,
		                      &nPixels,
							  sizeof(int),
							  cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_rasterMinMax,
		                      rasterMinMax,
							  2 * sizeof(double),
							  cudaMemcpyHostToDevice ) );

	/******************** Preparation for CUDA execution time recording ********************/

	cudaEvent_t timeStartEvent, timeEndEvent;

	HANDLE_ERROR( cudaEventCreate( &timeStartEvent, 0 ) );
	HANDLE_ERROR( cudaEventCreate( &timeEndEvent, 0 ) );

	HANDLE_ERROR( cudaEventRecord( timeStartEvent, 0 ) );

	/******************** ******************************************** ********************/

	if ( factorType == factor_Active ) 
	{
		normalizeActiveRasterPixel<<<128, 128>>>( dev_pixelMatrix, 
			                                      dev_nPixels, 
			                                      dev_rasterMinMax );
	}
	else
	{
		normalizeNegativeRasterPixel<<<128, 128>>>( dev_pixelMatrix, 
			                                        dev_nPixels, 
			                                        dev_rasterMinMax );
	}

	/********************** Check out CUDA execution time recording ***********************/
	HANDLE_ERROR( cudaEventRecord( timeEndEvent, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize(timeEndEvent) );

	float elapsedTime = 0;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, timeStartEvent, timeEndEvent ) );

	printf( "Time Consumption: %f ms. \n", elapsedTime );

	HANDLE_ERROR( cudaEventDestroy( timeStartEvent ) );
	HANDLE_ERROR( cudaEventDestroy( timeEndEvent ) );
	/******************** ******************************************** ********************/

	HANDLE_ERROR( cudaMemcpy( pixelMatrix,
		                      dev_pixelMatrix,
		                      nPixels * sizeof(float),
		                      cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaFree( dev_pixelMatrix ) );
	HANDLE_ERROR( cudaFree( dev_nPixels ) );
	HANDLE_ERROR( cudaFree( dev_rasterMinMax ) );
}

/**
 * <Interface>
 * Factor GeoTiff pixel normalization.
 * @param srcTifFile         ->  Source GeoTiff file path
 * @param outputTifFile      ->  Result output file path
 * @param factorType         ->  Evaluation factor type (Active/Negative)
 */
void geoTiffRasterPixelNormalization( const char    srcTifFile[],
	                                  const char    outputTifFile[],
                                      envFactorType factorType )
{
	float  *rasterPixels = NULL;
	int    tifWidth, tifLength;
	double *rasterMinMax;

	rasterMinMax = (double*)malloc( sizeof(double) * 2 );
	if ( rasterMinMax == NULL )
	{
		ERROR_INFO( "Out of memory" );
		return;
	}

	readTiffImageToMatrix( srcTifFile, 
		                   1, 
						   &rasterPixels );

	getTiffWidthLength( srcTifFile,
		                1,
						&tifWidth,
						&tifLength );

	getTiffMinMax( srcTifFile,
		           1,
				   rasterMinMax,
				   1 );

	rasterPixelNormalization( rasterPixels, 
	                          tifWidth, 
							  tifLength, 
					          rasterMinMax,
                              factorType );

	writeTiffImageRefSrc( outputTifFile, 
		                  srcTifFile, 
		                  1, 
		                  rasterPixels );

	double normalizedRasterMinMax[] = { 0, 1 };

	alterRasterMinMax( outputTifFile,
		               1,
	                   normalizedRasterMinMax );

	free(rasterMinMax);
}