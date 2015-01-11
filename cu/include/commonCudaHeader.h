#ifndef _COMMONCUDAHEADER_CUH_
#define _COMMONCUDAHEADER_CUH_

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

#endif