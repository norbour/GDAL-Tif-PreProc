#include <stdio.h>
#include <stdlib.h>	
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

#include "../include/book.h"
#include "../include/commCuda.h"
#include "../../include/tiffImageIO.h"  

/**
 * @Device
 * Superpose weighted factor raster pixels.
 */
__global__ void superposeTifRasters( float  *rasterPixels, 
	                                 double *factorWeightArray,
									 float  *outputPixelArray,
									 int    *nFactors,
									 int    *nPixels ) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < *nPixels)
	{
		if ( rasterPixels[tid] - 0xE0000000 != 0 ) // float nullPixelValue = 0xE0000000;
		{
			for ( int i = 0; i < *nFactors; i++ )
			{
				outputPixelArray[tid] += rasterPixels[tid + *nPixels * i] * factorWeightArray[i];
			}
		}

		tid += blockDim.x * gridDim.x; 
	}
}

/*
 * To check tiff resolution uniformity.
 * We except resolution(width & length) of all input tiff data are the same.
 * Get nPixels per raster data BTW
 */
int checkTifResUniformity( char  *factorTifNames[],
	                       int   nFiles,
						   int   *nPixels )
{
	if ( nFiles <= 0 )
	{
		ERROR_INFO( "The number of factor tiff files should be more than 1" );
		return 0;
	}

	int tifWidth,    tifLength;
	int preTifWidth, preTifLength;

	for ( int i = 0; i < nFiles; i++ )
	{
		getTiffWidthLength( factorTifNames[i],
							1,
							&tifWidth,
							&tifLength );
		if ( i != 0 && 
			( tifWidth != preTifWidth || tifLength != preTifLength ) )
		{
			ERROR_INFO( "Improper resolution size" );
			/*printf( "Improper resolution of %s  \n", factorTifNames[i] );*/
			return 0;
		}

		preTifWidth  = tifWidth;
		preTifLength = tifLength;
	}

	*nPixels = tifWidth * tifLength;

	return 1;
}

/*
 * To check tiff resolution uniformity.
 * We except resolution(width & length) of all input tiff data are the same.
 */
void getPackedPixelArray( char  *factorTifNames[],
	                      float **packedRasterPixels,
						  int   nPixels,
						  int   nFactors )
{
	*packedRasterPixels = (float*)malloc( nPixels * nFactors * sizeof(float) ); 
	if ( *packedRasterPixels == NULL)
	{
		ERROR_INFO( "Out of memory" );
		return;
	}

	float *rasterPixels = NULL;

	for ( int i = 0; i < nFactors; i++ )
	{ 
		readTiffImageToMatrix( factorTifNames[i], 
			                   1, 
			                   &rasterPixels );

		memcpy( *packedRasterPixels + (nPixels * i), 
			    rasterPixels, 
			    nPixels * sizeof(float) );

		CPLFree( rasterPixels ); // We use CPLMalloc() in readTiffImageToMatrix()
		rasterPixels = NULL;
	}
}

/**
 * <Core Function>
 * Weighted factor GeoTiff raster pixel superposition.
 * @param packedRasterPixels ->  pixel value array which packed all tiff file together
 * @param outputRasterPixels ->  superposition result raster pixel array
 * @param factorWeightArray  ->  factor weight array
 * @param nFactors           ->  number of factors
 * @param nPixels            ->  number of a single raster's pixels
 */
void rasterPixelSuperposition( float   *packedRasterPixels,
	                           float   **outputRasterPixels,
                               double  *factorWeightArray,
							   int     *nFactors, 
							   int     *nPixels              )
{
	float  *dev_packedRasterPixels = NULL; 
	double *dev_factorWeightArray  = NULL;
	int    *dev_nPixels            = NULL;
	int    *dev_nFactors           = NULL;
	float  *dev_outputRsterPixels  = NULL;     

	HANDLE_ERROR( cudaMalloc( (void**)&dev_packedRasterPixels, (*nPixels) * (*nFactors) * sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_factorWeightArray,  (*nFactors) * sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_nPixels,            sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_nFactors,           sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_outputRsterPixels,  (*nPixels) * sizeof(float) ) );

	HANDLE_ERROR( cudaMemcpy( dev_packedRasterPixels,
		                      packedRasterPixels,
		                      (*nPixels) * (*nFactors) * sizeof(float),
		                      cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_factorWeightArray,
		                      factorWeightArray,
		                      (*nFactors) * sizeof(double),
		                      cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_nPixels,
		                      nPixels,
		                      sizeof(int),
		                      cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_nFactors,
		                      nFactors,
		                      sizeof(int),
		                      cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemset( dev_outputRsterPixels, 
		                      0, 
		                      sizeof(float) * (*nPixels) ) );

	/******************** Preparation for CUDA execution time recording ********************/

	cudaEvent_t timeStartEvent, timeEndEvent;

	HANDLE_ERROR( cudaEventCreate( &timeStartEvent, 0 ) );
	HANDLE_ERROR( cudaEventCreate( &timeEndEvent, 0 ) );

	HANDLE_ERROR( cudaEventRecord( timeStartEvent, 0 ) );

	/******************** ******************************************** ********************/

	superposeTifRasters<<<128, 128>>>( dev_packedRasterPixels, 
		                               dev_factorWeightArray, 
		                               dev_outputRsterPixels,
		                               dev_nFactors,
		                               dev_nPixels            );

	/********************** Check out CUDA execution time recording ***********************/
	HANDLE_ERROR( cudaEventRecord( timeEndEvent, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize(timeEndEvent) );

	float elapsedTime = 0;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, timeStartEvent, timeEndEvent ) );

	printf( "Time Consumption: %f ms. \n", elapsedTime );

	HANDLE_ERROR( cudaEventDestroy( timeStartEvent ) );
	HANDLE_ERROR( cudaEventDestroy( timeEndEvent ) );
	/******************** ******************************************** ********************/

	*outputRasterPixels = (float*)malloc( (*nPixels) * sizeof(float) );

	HANDLE_ERROR( cudaMemcpy( *outputRasterPixels,
		                      dev_outputRsterPixels,
		                      (*nPixels) * sizeof(float),
		                      cudaMemcpyDeviceToHost ) );

	cudaFree( dev_packedRasterPixels );
	cudaFree( dev_factorWeightArray );
	cudaFree( dev_nPixels );
	cudaFree( dev_nFactors );
	cudaFree( dev_outputRsterPixels );
}

/**
 * <Interface>
 * Weighted factor GeoTiff raster pixel superposition.
 * @param factorTifNames     ->  A list of factor GeoTiff files
 * @param normalizedTifFile  ->  Result output file path
 * @param factorWeightArray  ->  factor weight array
 * @param nFactors           ->  number of factors
 */
void geoTiffRasterPixelSuperposition( char    *factorTifNames[],
	                                  char    *outputTifFile,
                                      double  *factorWeightArray,
									  int     nFactors)
{
	if ( nFactors <= 0 || factorTifNames     == NULL
		               || outputTifFile      == NULL
					   || factorWeightArray  == NULL )
	{
		ERROR_INFO( "Invalid number of factors input" );
		return;
	}

	int factorCount = nFactors;

	int nPixels;
	if ( !checkTifResUniformity( factorTifNames, nFactors, &nPixels ) )
		return;

	float *packedRasterPixels = NULL;
	getPackedPixelArray( factorTifNames,
	                     &packedRasterPixels,
						 nPixels, 
						 factorCount );

	float *outputRasterPixels = NULL;
	rasterPixelSuperposition( packedRasterPixels,
	                          &outputRasterPixels,
                              factorWeightArray,
							  &factorCount, 
							  &nPixels            );

    writeTiffImageRefSrc( outputTifFile, 
	                      factorTifNames[0], 
	                      1, 
					      outputRasterPixels );

   free( packedRasterPixels );
   free( outputRasterPixels );
}