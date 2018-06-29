#include <windows.h>
#include "nvapi.h"

int NVAPI::getCoreOC(int& deviceID)
{
	auto NvQueryInterface = (NvAPI_QueryInterface_t)GetProcAddress(LoadLibrary("nvapi64.dll"), (LPCSTR)"nvapi_QueryInterface");
	auto NvInit = (NvAPI_Initialize_t)NvQueryInterface(0x0150E828);
	auto NvUnload = (NvAPI_Unload_t)NvQueryInterface(0xD22BDD7E);
	auto NvEnumGPUs = (NvAPI_EnumPhysicalGPUs_t)NvQueryInterface(0xE5AC921F);
	auto NvGetPstates = (NvAPI_GPU_GetPstates20_t)NvQueryInterface(0x6FF81213);

	NvInit();

	int *hdlGPU[64] = { 0 };
	NvEnumGPUs(hdlGPU, &deviceID);

	NV_GPU_PERF_PSTATES20_INFO_V1 pstates_info;
	NvGetPstates(hdlGPU[0], &pstates_info);

	int coreOC{ ((pstates_info.pstates[0].clocks[0]).freqDelta_kHz.value) / 1000 };

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

	NvInit();

	int *hdlGPU[64] = { 0 };
	NvEnumGPUs(hdlGPU, &deviceID);

	NV_GPU_PERF_PSTATES20_INFO_V1 pstates_info;
	NvGetPstates(hdlGPU[0], &pstates_info);

	int memoryOC = { ((pstates_info.pstates[0].clocks[1]).freqDelta_kHz.value) / 1000 };

	NvUnload();

	return memoryOC;
}

bool NVAPI::foundNvAPI64()
{
	return (LoadLibrary("nvapi64.dll") != NULL);
}
