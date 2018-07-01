#include <windows.h>
#include "nvapi.h"

int NVAPI::getCoreOC(int& deviceID)
{
	auto NvQueryInterface = (NvAPI_QueryInterface_t)GetProcAddress(LoadLibrary("nvapi64.dll"), (LPCSTR)"nvapi_QueryInterface");
	auto NvInit = (NvAPI_Initialize_t)NvQueryInterface(0x0150E828);
	auto NvUnload = (NvAPI_Unload_t)NvQueryInterface(0xD22BDD7E);
	auto NvEnumGPUs = (NvAPI_EnumPhysicalGPUs_t)NvQueryInterface(0xE5AC921F);
	auto NvGetPstates = (NvAPI_GPU_GetPstates20_t)NvQueryInterface(0x6FF81213);
	auto NvGetFreq = (NvAPI_GPU_GetAllClockFrequencies_t)NvQueryInterface(0xDCB616C3);

	NvInit();

	int allGPUCount{ 0 };
	int *hdlGPU[64] = { 0 };
	NvEnumGPUs(hdlGPU, &allGPUCount);

	NV_GPU_PERF_PSTATES20_INFO_V1 pstatesInfo{ 0 };
	pstatesInfo.version = NV_GPU_PERF_PSTATES20_INFO_VER1;
	pstatesInfo.numPstates = 1;
	pstatesInfo.numClocks = 1;
	pstatesInfo.pstates[0].clocks[0].domainId = NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS;

	auto response{ (NvAPI_Status)NvGetPstates(hdlGPU[deviceID], &pstatesInfo) };
	if (response != NVAPI::NVAPI_OK)
	{
		std::cout << "[ERROR] NVML response ID: " << response << "\n";
		return 0;
	}

	int coreOC{ ((pstatesInfo.pstates[0].clocks[0]).freqDelta_kHz.value) / 1000 };

	NvUnload();

	return coreOC;
}

int NVAPI::getMemoryOC(int& deviceID)
{
	auto NvQueryInterface = (NvAPI_QueryInterface_t)GetProcAddress(LoadLibrary("nvapi64.dll"), (LPCSTR)"nvapi_QueryInterface");
	auto NvInit = (NvAPI_Initialize_t)NvQueryInterface(0x0150E828);
	auto NvUnload = (NvAPI_Unload_t)NvQueryInterface(0xD22BDD7E);
	auto NvEnumGPUs = (NvAPI_EnumPhysicalGPUs_t)NvQueryInterface(0xE5AC921F);
	auto NvGetPstates = (NvAPI_GPU_GetPstates20_t)NvQueryInterface(0x6FF81213);
	auto NvGetFreq = (NvAPI_GPU_GetAllClockFrequencies_t)NvQueryInterface(0xDCB616C3);

	NvInit();

	int allGPUCount = { 0 };
	int *hdlGPU[64] = { 0 };
	NvEnumGPUs(hdlGPU, &allGPUCount);

	NV_GPU_PERF_PSTATES20_INFO_V1 pstatesInfo{ 0 };
	pstatesInfo.version = NV_GPU_PERF_PSTATES20_INFO_VER1;
	pstatesInfo.numPstates = 1;
	pstatesInfo.numClocks = 1;
	pstatesInfo.pstates[0].clocks[0].domainId = NVAPI_GPU_PUBLIC_CLOCK_MEMORY;

	auto response{ (NvAPI_Status)NvGetPstates(hdlGPU[deviceID], &pstatesInfo) };
	if (response != NVAPI::NVAPI_OK)
	{
		std::cout << "[ERROR] NVML response ID: " << response << "\n";
		return 0;
	}

	int memoryOC{ ((pstatesInfo.pstates[0].clocks[1]).freqDelta_kHz.value) / 1000 };

	NvUnload();

	return memoryOC;
}

bool NVAPI::foundNvAPI64()
{
	return (LoadLibrary("nvapi64.dll") != NULL);
}
