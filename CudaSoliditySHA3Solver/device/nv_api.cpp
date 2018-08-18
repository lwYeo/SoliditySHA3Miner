#include <windows.h>
#include "nv_api.h"

#pragma unmanaged

// --------------------------------------------------------------------
// Static
// --------------------------------------------------------------------

NV_API::QueryInterface_t NV_API::QueryInterface{ NULL };
NV_API::GetErrorMessage_t NV_API::GetErrorMessage{ NULL };

NV_API::Initialize_t NV_API::Initialize{ NULL };
NV_API::Unload_t NV_API::Unload{ NULL };
NV_API::EnumPhysicalGPUs_t NV_API::EnumPhysicalGPUs{ NULL };
NV_API::GPU_GetBusID_t NV_API::GPU_GetBusID{ NULL };

NV_API::GPU_GetPstates20_t NV_API::GPU_GetPstates20{ NULL };
NV_API::GPU_GetAllClockFrequencies_t NV_API::GPU_GetAllClockFrequencies{ NULL };
NV_API::DLL_ClientPowerPoliciesGetStatus_t NV_API::DLL_ClientPowerPoliciesGetStatus{ NULL };
NV_API::DLL_ClientThermalPoliciesGetLimit_t NV_API::DLL_ClientThermalPoliciesGetLimit{ NULL };
NV_API::GPU_GetCoolersSettings_t NV_API::GPU_GetCoolersSettings{ NULL };

NV_API::GPU_GetTachReading_t NV_API::GPU_GetTachReading{ NULL };
NV_API::GPU_GetThermalSettings_t NV_API::GPU_GetThermalSettings{ NULL };
NV_API::GPU_GetCurrentPstate_t NV_API::GPU_GetCurrentPstate{ NULL };
NV_API::GPU_GetDynamicPstatesInfoEx_t NV_API::GPU_GetDynamicPstatesInfoEx{ NULL };
NV_API::GPU_GetPerfDecreaseInfo_t NV_API::GPU_GetPerfDecreaseInfo{ NULL };

bool NV_API::isInitialized{ false };
NvPhysicalGpuHandle NV_API::gpuHandles[NVAPI_MAX_PHYSICAL_GPUS];
NvU32 NV_API::gpuCount{ 0 };

bool NV_API::foundNvAPI64()
{
	return (LoadLibrary(NvAPI64) != NULL);
}

void NV_API::initialize()
{
	if (isInitialized) return;

	QueryInterface = (QueryInterface_t)GetProcAddress(LoadLibrary(NvAPI64), NvAPI_QueryInterface);
	if (QueryInterface == NULL) throw std::exception("Failed to initialize NvAPI64.");

	GetErrorMessage = (GetErrorMessage_t)QueryInterface(NvAPI_FUNCTIONS::GetErrorMessage);

	Initialize = (Initialize_t)QueryInterface(NvAPI_FUNCTIONS::Initialize);
	Unload = (Unload_t)QueryInterface(NvAPI_FUNCTIONS::Unload);
	EnumPhysicalGPUs = (EnumPhysicalGPUs_t)QueryInterface(NvAPI_FUNCTIONS::EnumPhysicalGPUs);
	GPU_GetBusID = (GPU_GetBusID_t)QueryInterface(NvAPI_FUNCTIONS::GetBusID);

	GPU_GetAllClockFrequencies = (GPU_GetAllClockFrequencies_t)QueryInterface(NvAPI_FUNCTIONS::GetAllClockFrequencies);
	DLL_ClientPowerPoliciesGetStatus = (DLL_ClientPowerPoliciesGetStatus_t)QueryInterface(NvAPI_FUNCTIONS::ClientPowerPoliciesGetStatus);
	DLL_ClientThermalPoliciesGetLimit = (DLL_ClientThermalPoliciesGetLimit_t)QueryInterface(NvAPI_FUNCTIONS::ClientThermalPoliciesGetLimit);
	GPU_GetCoolersSettings = (GPU_GetCoolersSettings_t)QueryInterface(NvAPI_FUNCTIONS::GetCoolersSettings);

	GPU_GetTachReading = (GPU_GetTachReading_t)QueryInterface(NvAPI_FUNCTIONS::GetTachReading);
	GPU_GetThermalSettings = (GPU_GetThermalSettings_t)QueryInterface(NvAPI_FUNCTIONS::GetThermalSettings);
	GPU_GetPstates20 = (GPU_GetPstates20_t)QueryInterface(NvAPI_FUNCTIONS::GetPstates20);
	GPU_GetDynamicPstatesInfoEx = (GPU_GetDynamicPstatesInfoEx_t)QueryInterface(NvAPI_FUNCTIONS::GetDynamicPstatesInfoEx);
	GPU_GetCurrentPstate = (GPU_GetCurrentPstate_t)QueryInterface(NvAPI_FUNCTIONS::GetCurrentPstate);
	GPU_GetPerfDecreaseInfo = (GPU_GetPerfDecreaseInfo_t)QueryInterface(NvAPI_FUNCTIONS::GetPerfDecreaseInfo);

	isInitialized = (Initialize() == NVAPI_OK);

	EnumPhysicalGPUs(gpuHandles, &gpuCount);
}

void NV_API::unload()
{
	if (QueryInterface == NULL || Unload == NULL) return;

	gpuCount = -1;
	isInitialized = false;

	Unload();
	if (gpuHandles != NULL) free(gpuHandles);
}

// --------------------------------------------------------------------
// Public
// --------------------------------------------------------------------

void NV_API::assignPciBusID(uint32_t pciBusID)
{
	NvU32 tempBusID;
	deviceBusID = pciBusID;
	deviceHandle = NULL;

	for (NvU32 h{ 0 }; h < gpuCount; ++h)
	{
		auto status = GPU_GetBusID(gpuHandles[h], &tempBusID);

		if (status == NVAPI_OK && tempBusID == NvU32{ pciBusID })
			deviceHandle = gpuHandles[h];
	}
}

NvAPI_Status NV_API::getErrorMessage(NvAPI_Status errStatus, std::string *message)
{
	NvAPI_ShortString errorMessage;

	auto status = GetErrorMessage(errStatus, errorMessage);
	if (status != NVAPI_OK) return status;

	*message = std::string{ errorMessage };

	return NVAPI_OK;
}

NvAPI_Status NV_API::getSettingMaxCoreClock(int *maxCoreClock)
{
	*maxCoreClock = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_PERF_PSTATES20_INFO_V1 pstatesInfo{ 0 };
	pstatesInfo.version = NV_GPU_PERF_PSTATES20_INFO_VER_1;

	auto status = GPU_GetPstates20(deviceHandle, &pstatesInfo);
	if (status != NVAPI_OK) return status;

	*maxCoreClock = (int)((pstatesInfo.pstates[0].clocks[0]).data.range.maxFreq_kHz / 1000ul);
	return NVAPI_OK;
}

NvAPI_Status NV_API::getSettingMaxMemoryClock(int *maxMemoryClock)
{
	*maxMemoryClock = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_PERF_PSTATES20_INFO_V1 pstatesInfo{ 0 };
	pstatesInfo.version = NV_GPU_PERF_PSTATES20_INFO_VER_1;

	auto status = GPU_GetPstates20(deviceHandle, &pstatesInfo);
	if (status != NVAPI_OK) return status;

	*maxMemoryClock = (int)((pstatesInfo.pstates[0].clocks[1]).data.single.freq_kHz / 1000ul);

	return NVAPI_OK;
}

NvAPI_Status NV_API::getSettingPowerLimit(int *powerLimit)
{
	*powerLimit = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NVAPI_GPU_POWER_STATUS powerPolicy{ 0 };
	powerPolicy.version = NVAPI_GPU_POWER_STATUS_VER;

	auto status = DLL_ClientPowerPoliciesGetStatus(deviceHandle, &powerPolicy);
	if (status != NVAPI_OK) return status;

	*powerLimit = (int)(powerPolicy.entries[0].power / 1000ul);

	return NVAPI_OK;
}

NvAPI_Status NV_API::getSettingThermalLimit(int *thermalLimit)
{
	*thermalLimit = INT32_MIN;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NVAPI_GPU_THERMAL_LIMIT_V2 thermalPolicy{ 0 };
	thermalPolicy.version = NVAPI_GPU_THERMAL_LIMIT_VER_2;

	auto status = DLL_ClientThermalPoliciesGetLimit(deviceHandle, &thermalPolicy);
	if (status != NVAPI_OK) return status;

	*thermalLimit = thermalPolicy.entries[0].value >> 8;

	return NVAPI_OK;
}

NvAPI_Status NV_API::getSettingFanLevelPercent(int *fanLevel)
{
	*fanLevel = INT32_MIN;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_COOLER_SETTINGS_V2 coolerSetting{ 0 };
	coolerSetting.version = NV_GPU_COOLER_SETTINGS_VER_2;

	auto status = GPU_GetCoolersSettings(deviceHandle, 0ul, &coolerSetting);
	if (status != NVAPI_OK) return status;

	*fanLevel = coolerSetting.cooler[0].currentLevel;

	return NVAPI_OK;
}

NvAPI_Status NV_API::getCurrentFanTachometerRPM(int *tachometerRPM)
{
	*tachometerRPM = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	auto status = GPU_GetTachReading(deviceHandle, (NvU32 *)tachometerRPM);

	return status;
}

NvAPI_Status NV_API::getCurrentTemperature(int *temperature)
{
	*temperature = INT32_MIN;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_THERMAL_SETTINGS_V2 currentGpuCoreThermal{ 0 };
	currentGpuCoreThermal.version = NV_GPU_THERMAL_SETTINGS_VER_2;

	auto status = GPU_GetThermalSettings(deviceHandle, NVAPI_THERMAL_TARGET_NONE, &currentGpuCoreThermal);
	if (status != NVAPI_OK) return status;

	*temperature = (int)currentGpuCoreThermal.sensor[0].currentTemp;

	return NVAPI_OK;
}

NvAPI_Status NV_API::getCurrentCoreClock(int *coreClock)
{
	*coreClock = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_CLOCK_FREQUENCIES_V2 clockTable{ 0 };
	clockTable.version = NV_GPU_CLOCK_FREQUENCIES_VER_2;
	clockTable.ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;

	auto status = GPU_GetAllClockFrequencies(deviceHandle, &clockTable);
	if (status != NVAPI_OK) return status;

	*coreClock = (clockTable.domain[NVAPI_GPU_PUBLIC_CLOCK_PROCESSOR].bIsPresent)
		? (int)(clockTable.domain[NVAPI_GPU_PUBLIC_CLOCK_PROCESSOR].frequency / 1000ul)
		: (int)(clockTable.domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency / 1000ul);

	return NVAPI_OK;
}

NvAPI_Status NV_API::getCurrentMemoryClock(int *memoryClock)
{
	*memoryClock = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_CLOCK_FREQUENCIES_V2 clockTable{ 0 };
	clockTable.version = NV_GPU_CLOCK_FREQUENCIES_VER_2;
	clockTable.ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;

	auto status = GPU_GetAllClockFrequencies(deviceHandle, &clockTable);
	if (status != NVAPI_OK) return status;

	*memoryClock = (int)(clockTable.domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency / 1000ul);

	return NVAPI_OK;
}

NvAPI_Status NV_API::getCurrentUtilizationPercent(int *utilization)
{
	*utilization = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_DYNAMIC_PSTATES_INFO_EX dynPstateInfo{ 0 };
	dynPstateInfo.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;

	auto status = GPU_GetDynamicPstatesInfoEx(deviceHandle, &dynPstateInfo);
	if (status != NVAPI_OK) return status;

	*utilization = (int)dynPstateInfo.utilization[0].percentage;

	return NVAPI_OK;
}

NvAPI_Status NV_API::getCurrentPstate(int *pstate)
{
	*pstate = -1;
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NV_GPU_PERF_PSTATE_ID currentPstate{ NVAPI_GPU_PERF_PSTATE_UNDEFINED };

	auto status = GPU_GetCurrentPstate(deviceHandle, &currentPstate);
	if (status != NVAPI_OK) return status;

	*pstate = (int)currentPstate;

	return NVAPI_OK;
}

NvAPI_Status NV_API::getCurrentThrottleReasons(std::string *reasons)
{
	*reasons = "";
	if (deviceHandle == NULL) return NVAPI_NVIDIA_DEVICE_NOT_FOUND;

	NVAPI_GPU_PERF_DECREASE perfDescInfo{ NV_GPU_PERF_DECREASE_NONE };

	auto status = GPU_GetPerfDecreaseInfo(deviceHandle, &perfDescInfo);
	if (status != NVAPI_OK) return status;

	if (perfDescInfo == NV_GPU_PERF_DECREASE_NONE) *reasons = "";
	else if (perfDescInfo == NV_GPU_PERF_DECREASE_REASON_UNKNOWN) *reasons = "Unknown";
	else
	{
		*reasons = "";
		if ((perfDescInfo & NV_GPU_PERF_DECREASE_REASON_THERMAL_PROTECTION) == NV_GPU_PERF_DECREASE_REASON_THERMAL_PROTECTION)
			*reasons += reasons->empty() ? "Thermal protection" : ", Thermal protection";

		if ((perfDescInfo & NV_GPU_PERF_DECREASE_REASON_POWER_CONTROL) == NV_GPU_PERF_DECREASE_REASON_POWER_CONTROL)
			*reasons += reasons->empty() ? "Power cap" : ", Power cap";

		if ((perfDescInfo & NV_GPU_PERF_DECREASE_REASON_AC_BATT) == NV_GPU_PERF_DECREASE_REASON_AC_BATT)
			*reasons += reasons->empty() ? "Battery power" : ", Battery power";

		if ((perfDescInfo & NV_GPU_PERF_DECREASE_REASON_API_TRIGGERED) == NV_GPU_PERF_DECREASE_REASON_API_TRIGGERED)
			*reasons += reasons->empty() ? "Application setting" : ", Application setting";

		if ((perfDescInfo & NV_GPU_PERF_DECREASE_REASON_INSUFFICIENT_POWER) == NV_GPU_PERF_DECREASE_REASON_INSUFFICIENT_POWER)
			*reasons += reasons->empty() ? "Insufficient power" : ", Insufficient power";
	}

	return NVAPI_OK;
}
