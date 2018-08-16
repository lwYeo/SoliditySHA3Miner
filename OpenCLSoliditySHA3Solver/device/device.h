#pragma once

#define DEFAULT_INTENSITY 24.223f
#define DEFAULT_LOCAL_WORK_SIZE 128u
#define MAX_SOLUTION_COUNT_DEVICE 32u

#define KERNEL_FILE "sha3Kernel.cl"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <string.h>
#include "adl_api.h"
#include "../types.h"

#pragma managed(push, off)

#ifdef _M_CEE 
#	undef _M_CEE 
#	include <thread>

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
#endif

#	define _M_CEE 001 
#else 
#	include <thread>

#if defined(__APPLE__) || defined(__MACOSX)
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
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

	bool checkChanges;
	bool isNewTarget;
	bool isNewMessage;

	state_t currentMidState;
	uint64_t currentHigh64Target[1];

	std::vector<size_t> maxWorkItemSizes;
	size_t maxWorkGroupSize;
	cl_uint maxComputeUnits;
	cl_ulong maxMemAllocSize;
	cl_ulong globalMemSize;

	size_t localWorkSize;
	size_t globalWorkSize;

	uint32_t *h_solutionCount;
	uint64_t *h_solutions;

	cl_mem solutionCountBuffer;
	cl_mem solutionsBuffer;
	cl_mem midstateBuffer;
	cl_mem targetBuffer;

	cl_command_queue queue;
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	
	cl_event kernelWaitEvent;
	uint32_t kernelWaitSleepDuration;

private:
	int pciBusID;
	ADL_API m_api;
	uint32_t computeCapability;

public:
	Device(int devEnum, cl_device_id devID, cl_device_type devType, cl_platform_id devPlatformID, float const userDefIntensity = 0, uint32_t userLocalWorkSize = 0);

	bool isAPP();
	bool isCUDA();
	bool isINTEL();

	bool getSettingMaxCoreClock(int *maxCoreClock, std::string *errorMessage);
	bool getSettingMaxMemoryClock(int *maxMemoryClock, std::string *errorMessage);
	bool getSettingPowerLimit(int *powerLimit, std::string *errorMessage);
	bool getSettingThermalLimit(int *thermalLimit, std::string *errorMessage);
	bool getSettingFanLevelPercent(int *fanLevel, std::string *errorMessage);

	bool getCurrentFanTachometerRPM(int *tachometerRPM, std::string *errorMessage);
	bool getCurrentTemperature(int *temperature, std::string *errorMessage);
	bool getCurrentCoreClock(int *coreClock, std::string *errorMessage);
	bool getCurrentMemoryClock(int *memoryClock, std::string *errorMessage);
	bool getCurrentUtilizationPercent(int *utilization, std::string *errorMessage);

	uint64_t hashRate();

	void initialize(std::string& errorMessage);
	void setIntensity(float const intensity);

private:
	bool setKernelArgs(std::string& errorMessage);
};
