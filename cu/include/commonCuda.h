#ifndef _COMMCUDA_CUH_
#define _COMMCUDA_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Environment evaluation factor type (Active/Negative)
 */
enum envFactorType
{
    factor_Negative  =  0,     /** Negative factor */
    factor_Active    =  1      /** Active factor   */
};

/**
 * Collect all the information of every CUDA device on this computer 
 * and return as cudaDeviceProp array.
 */
extern cudaDeviceProp* getCudaDevicesInfo();

/**
 * <Interface>
 * Factor GeoTiff pixel normalization.
 * @param srcTifFile         ->  Source GeoTiff file path
 * @param normalizedTifFile  ->  raster width
 * @param factorType         -> evaluation factor type (Active/Negative)
 */
extern void geoTiffRasterPixelNormalization( const char    srcTifFile[],
	                                         const char    outputTifFile[],
                                             envFactorType factorType );

/**
 * <Interface>
 * Weighted factor GeoTiff raster pixel superposition.
 * @param factorTifNames     ->  A list of factor GeoTiff files
 * @param normalizedTifFile  ->  Result output file path
 * @param factorType         ->  Evaluation factor type (Active/Negative)
 */
extern void geoTiffRasterPixelSuperposition( char    *factorTifNames[],
	                                         char    *outputTifFile,
                                             double  *factorWeightArray,
									         int     nFactors );

#endif