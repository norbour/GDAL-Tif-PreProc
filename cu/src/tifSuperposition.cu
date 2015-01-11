#include <stdio.h>
#include <stdlib.h>	
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/book.h"
#include "../include/commonCudaHeader.h"
#include "../../include/tiffImageIO.h"

/**
 * @Device
 * Active factor GeoTiff pixel normalization.
 */
__global__ void superposeTifRasters( char   *factorTifNames[], 
	                                 double *nPixels, 
								     int    *factorNum ) 
{

}