#include <stdio.h>
#include <stdlib.h>	

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/book.h"
#include "../include/commonCudaHeader.h"

/*
 * Collect all the information of every CUDA device on this computer 
 * and return as cudaDeviceProp array.
 */
cudaDeviceProp* getCudaDevicesInfo() {
	cudaDeviceProp* deviceInfos = NULL;
	cudaDeviceProp prop;

	int deviceCount;
	HANDLE_ERROR( cudaGetDeviceCount( &deviceCount ) );
	deviceInfos = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * deviceCount);

	for (int i=0; i < deviceCount; i++)
	{
		HANDLE_ERROR( cudaGetDeviceProperties( &prop, i) );
		memcpy(deviceInfos + i, &prop, sizeof(cudaDeviceProp));
		printf( "   --- General Information for device %d ---   \n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );
		printf( "Device copy overlap: " );
		if (prop.deviceOverlap)
		{
			printf( "Enabled\n" );
		} 
		else
		{
			printf( "Disabled\n" );
		}
		printf( "Kernel execution timeout : " );
		if (prop.kernelExecTimeoutEnabled)
		{
			printf( "Enabled\n" );
		} 
		else
		{
			printf( "Disabled\n" );
		}

		printf( "   --- Memory Information for device %d ---   \n", i );
		printf( "Total global memory: %ld\n", prop.totalGlobalMem );
		printf( "Total constant memory: %ld\n", prop.totalConstMem );
		printf( "Max mem pitch: %ld\n", prop.memPitch );
		printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

		printf( "   --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count:  %d\n",
			prop.multiProcessorCount );
		printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp:  %d\n", prop.regsPerBlock );
		printf( "Threads in warp:  %d\n", prop.warpSize );
		printf( "Max threads per block:  %d\n",
			prop.maxThreadsPerBlock );
		printf( "Max thread dimensions:  (%d, %d, %d)\n",
			prop.maxThreadsDim[0], prop.maxThreadsDim[1],
			prop.maxThreadsDim[2] );
		printf( "Max grid dimensions:  (%d, %d, %d)\n",
			prop.maxGridSize[0], prop.maxGridSize[1],
			prop.maxGridSize[2] );
		printf( "\n" );
	}

	return deviceInfos;
}