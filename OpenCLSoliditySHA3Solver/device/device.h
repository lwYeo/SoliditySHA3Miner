#pragma once

#define DEFAULT_INTENSITY 25.0F
#define DEFAULT_LOCAL_WORK_SIZE 128
#define MAX_SOLUTION_COUNT_DEVICE 32

#define KERNEL_FILE "sha3Kernel.cl"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <algorithm>
#include <atomic>
#include <fstream>
#include <string.h>
#include "../types.h"

#pragma managed(push, off)

#ifdef _M_CEE 
#	undef _M_CEE 
#	include <mutex>

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
//#	include <OpenCL/opencl.h>
#else
#	include <CL/cl.hpp>
//#	include <CL/opencl.h>
#endif

#	define _M_CEE 001 
#else 
#	include <mutex>

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
//#	include <OpenCL/opencl.h>
#else
#	include <CL/cl.hpp>
//#	include <CL/opencl.h>
#endif

#endif 

#pragma managed(pop)

class Device
{
public:
	static std::vector<std::unique_ptr<Device>> devices;
	static const char *kernelSource;
	static size_t kernelSourceSize;

	template<typename T>
	static const char* getOpenCLErrorCodeStr(T &input);

	static bool preInitialize(std::string& errorMessage);

public:
	int deviceEnum;
	cl_device_id deviceID;
	cl_device_type deviceType;
	cl_platform_id platformID;
	cl_int status;

	float userDefinedIntensity;
	bool initialized;
	bool mining;

	std::thread miningThread;
	std::atomic<uint64_t> hashCount;
	std::atomic<std::chrono::steady_clock::time_point> hashStartTime;

	std::string platformName;
	std::string openCLVersion;
	std::string vendor;
	std::string name;
	std::string extensions;

	std::vector<size_t> maxWorkItemSizes;
	size_t maxWorkGroupSize;
	cl_uint maxComputeUnits;
	cl_ulong maxMemAllocSize;
	cl_ulong globalMemSize;

	size_t localWorkSize;
	size_t globalWorkSize;

	uint64_t *h_Solutions;

	cl_mem solutionsBuffer;
	cl_mem midstateBuffer;

	cl_command_queue queue;
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	
	cl_event kernelWaitEvent;
	uint32_t kernelWaitSleepDuration;

private:
	uint32_t computeCapability;
	std::mutex hashRateMutex;

public:
	Device(int devEnum, cl_device_id devID, cl_device_type devType, cl_platform_id devPlatformID, float const userDefIntensity = 0, uint32_t userLocalWorkSize = 0);

	bool isAPP();
	bool isCUDA();
	bool isINTEL();

	uint64_t hashRate();

	void initialize(std::string& errorMessage);
	void setIntensity(float const intensity);
};
