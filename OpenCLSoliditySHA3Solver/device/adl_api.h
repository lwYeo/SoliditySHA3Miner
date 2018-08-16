#pragma once
#include <string>
#include <Windows.h>
#include "adl_include/adl_sdk.h"

#define ADL64_API										"atiadlxx.dll"

class ADL_API
{
private:
	typedef int(*ADL_MAIN_CONTROL_CREATE)				(ADL_MAIN_MALLOC_CALLBACK, int);
	typedef int(*ADL_MAIN_CONTROL_DESTROY)				();
	typedef int(*ADL_FLUSH_DRIVER_DATA)					(int);
	typedef int(*ADL2_ADAPTER_ACTIVE_GET)				(ADL_CONTEXT_HANDLE, int, int *);

	typedef int(*ADL_ADAPTER_NUMBEROFADAPTERS_GET)		(int*);
	typedef int(*ADL_ADAPTER_ADAPTERINFO_GET)			(LPAdapterInfo, int);
	typedef int(*ADL_ADAPTERX2_CAPS)					(int, int*);
	typedef int(*ADL2_OVERDRIVE_CAPS)					(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int *iSupported, int *iEnabled, int *iVersion);
	typedef int(*ADL2_OVERDRIVEN_CAPABILITIES_GET)		(ADL_CONTEXT_HANDLE, int, ADLODNCapabilities *);
	typedef int(*ADL2_OVERDRIVEN_SYSTEMCLOCKS_GET)		(ADL_CONTEXT_HANDLE, int, ADLODNPerformanceLevels *);
	typedef int(*ADL2_OVERDRIVEN_SYSTEMCLOCKS_SET)		(ADL_CONTEXT_HANDLE, int, ADLODNPerformanceLevels *);
	typedef int(*ADL2_OVERDRIVEN_MEMORYCLOCKS_GET)		(ADL_CONTEXT_HANDLE, int, ADLODNPerformanceLevels *);
	typedef int(*ADL2_OVERDRIVEN_MEMORYCLOCKS_SET)		(ADL_CONTEXT_HANDLE, int, ADLODNPerformanceLevels *);
	typedef int(*ADL2_OVERDRIVEN_PERFORMANCESTATUS_GET)	(ADL_CONTEXT_HANDLE, int, ADLODNPerformanceStatus *);
	typedef int(*ADL2_OVERDRIVEN_FANCONTROL_GET)		(ADL_CONTEXT_HANDLE, int, ADLODNFanControl *);
	typedef int(*ADL2_OVERDRIVEN_FANCONTROL_SET)		(ADL_CONTEXT_HANDLE, int, ADLODNFanControl *);
	typedef int(*ADL2_OVERDRIVEN_POWERLIMIT_GET)		(ADL_CONTEXT_HANDLE, int, ADLODNPowerLimitSetting *);
	typedef int(*ADL2_OVERDRIVEN_POWERLIMIT_SET)		(ADL_CONTEXT_HANDLE, int, ADLODNPowerLimitSetting *);
	typedef int(*ADL2_OVERDRIVEN_TEMPERATURE_GET)		(ADL_CONTEXT_HANDLE, int, int, int *);

	static HINSTANCE									hDLL;

	static ADL_MAIN_CONTROL_CREATE						ADL_Main_Control_Create;
	static ADL_MAIN_CONTROL_DESTROY						ADL_Main_Control_Destroy;
	static ADL_ADAPTER_NUMBEROFADAPTERS_GET				ADL_Adapter_NumberOfAdapters_Get;
	static ADL_ADAPTER_ADAPTERINFO_GET					ADL_Adapter_AdapterInfo_Get;
	static ADL_ADAPTERX2_CAPS							ADL_AdapterX2_Caps;
	static ADL2_ADAPTER_ACTIVE_GET						ADL2_Adapter_Active_Get;
	static ADL2_OVERDRIVEN_CAPABILITIES_GET				ADL2_OverdriveN_Capabilities_Get;
	static ADL2_OVERDRIVEN_SYSTEMCLOCKS_GET				ADL2_OverdriveN_SystemClocks_Get;
	static ADL2_OVERDRIVEN_SYSTEMCLOCKS_SET				ADL2_OverdriveN_SystemClocks_Set;
	static ADL2_OVERDRIVEN_PERFORMANCESTATUS_GET		ADL2_OverdriveN_PerformanceStatus_Get;
	static ADL2_OVERDRIVEN_FANCONTROL_GET				ADL2_OverdriveN_FanControl_Get;
	static ADL2_OVERDRIVEN_FANCONTROL_SET				ADL2_OverdriveN_FanControl_Set;
	static ADL2_OVERDRIVEN_POWERLIMIT_GET				ADL2_OverdriveN_PowerLimit_Get;
	static ADL2_OVERDRIVEN_POWERLIMIT_SET				ADL2_OverdriveN_PowerLimit_Set;
	static ADL2_OVERDRIVEN_MEMORYCLOCKS_GET				ADL2_OverdriveN_MemoryClocks_Get;
	static ADL2_OVERDRIVEN_MEMORYCLOCKS_GET				ADL2_OverdriveN_MemoryClocks_Set;
	static ADL2_OVERDRIVE_CAPS							ADL2_Overdrive_Caps;
	static ADL2_OVERDRIVEN_TEMPERATURE_GET				ADL2_OverdriveN_Temperature_Get;

	static int											numberOfAdapters;
	static LPAdapterInfo								lpAdapterInfo;

	static void* __stdcall								ADL_Main_Memory_Alloc(int iSize);	// Memory allocation function

	uint32_t m_adapterBusID;
	AdapterInfo m_adapterInfo;

	ADL_CONTEXT_HANDLE m_context;
	int m_supported;
	int m_enabled;
	int m_version;

	bool checkVersion(std::string *errorMessage);
	bool getOverDriveNCapabilities(ADLODNCapabilities *capabilities, std::string *errorMessage);

public:
	static bool isInitialized;
	
	static bool foundAdlApi();
	static void initialize();
	static void unload();

	void assignPciBusID(int adapterBusID);

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
};
