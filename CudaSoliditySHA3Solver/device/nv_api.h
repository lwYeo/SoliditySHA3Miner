/*
   Copyright 2018 Lip Wee Yeo Amano

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <iostream>
#include <stdexcept>
#include <cstring>
#include <string>
#include "nv_apiEnum.h"
#include "../types.h"

#define NvAPI64										"nvapi64.dll"
#ifdef __linux__
#	define	NvAPI_QueryInterface					"nvapi_QueryInterface"
#else
#	define	NvAPI_QueryInterface					(LPCSTR)"nvapi_QueryInterface"
#endif

#define NVAPI_SHORT_STRING_MAX						64
#define NVAPI_MAX_PHYSICAL_GPUS						64
#define NVAPI_MAX_GPU_UTILIZATIONS					8
#define NVAPI_MAX_GPU_PSTATE20_PSTATES				16
#define NVAPI_MAX_GPU_PSTATE20_CLOCKS				8
#define NVAPI_MAX_GPU_PSTATE20_BASE_VOLTAGES		4
#define NVAPI_MAX_THERMAL_SENSORS_PER_GPU			3
#define NVAPI_MAX_COOLERS_PER_GPU					20

#ifndef __unix

// mac os 32-bit still needs this
#if ( (defined(macintosh) && defined(__LP64__) && (__NVAPI_RESERVED0__)) || (!defined(macintosh) && defined(__NVAPI_RESERVED0__)) )

typedef unsigned int								NvU32;				// 0 to 4294967295
typedef signed int									NvS32;				// -2147483648 to 2147483647

#else

typedef unsigned long								NvU32;				// 0 to 4294967295
typedef signed long									NvS32;				// -2147483648 to 2147483647

#endif

#else

typedef unsigned int								NvU32;				// 0 to 4294967295
typedef signed int									NvS32;				// -2147483648 to 2147483647

#endif

typedef char										NvAPI_ShortString[NVAPI_SHORT_STRING_MAX];

// =======================================================================================================
//! NVAPI Handles - These handles are retrieved from various calls and passed in to others in NvAPI
//!                 These are meant to be opaque types.  Do not assume they correspond to indices, HDCs,
//!                 display indexes or anything else.
//!                 Most handles remain valid until a display re-configuration (display mode set) or GPU
//!                 reconfiguration (going into or out of SLI modes) occurs.  If NVAPI_HANDLE_INVALIDATED
//!                 is received by an app, it should discard all handles, and re-enumerate them.
// =======================================================================================================
#define NV_DECLARE_HANDLE(name)	\
	struct name##__ { int unused; };	\
	typedef struct name##__ *name

NV_DECLARE_HANDLE(NvPhysicalGpuHandle);									//!< A single physical GPU

// =========================================================================================
//!  NvAPI Version Definition \n
//!  Maintain per structure specific version define using the MAKE_NVAPI_VERSION macro. \n
//!  Usage: #define NV_GENLOCK_STATUS_VER  MAKE_NVAPI_VERSION(NV_GENLOCK_STATUS, 1)
//!  \ingroup nvapitypes
// =========================================================================================
#define MAKE_NVAPI_VERSION(typeName, ver)			(NvU32)(sizeof(typeName) | ((ver) << 16))

class NV_API
{
private:
	typedef struct _NV_GPU_PERF_PSTATES20_PARAM_DELTA					//! Used to describe both voltage and frequency deltas
	{
		NvS32										value;				//! Value of parameter delta (in respective units [kHz, uV])
		struct
		{
			NvS32									min;				//! Min value allowed for parameter delta (in respective units [kHz, uV])
			NvS32									max;				//! Max value allowed for parameter delta (in respective units [kHz, uV])
		} valueRange;
	} NV_GPU_PERF_PSTATES20_PARAM_DELTA;

	typedef struct _NV_GPU_PSTATE20_CLOCK_ENTRY_V1						//! Used to describe single clock entry
	{
		NV_GPU_PUBLIC_CLOCK_ID						domainId;			//! ID of the clock domain

		NV_GPU_PERF_PSTATE20_CLOCK_TYPE_ID			typeId;				//! Clock type ID
		NvU32										bIsEditable : 1;
		NvU32										reserved : 31;		//! These bits are reserved for future use (must be always 0)
		NV_GPU_PERF_PSTATES20_PARAM_DELTA			freqDelta_kHz;		//! Current frequency delta from nominal settings in (kHz)

		union _data														//! Clock domain type dependant information
		{
			struct _single
			{
				NvU32								freq_kHz;			//! Clock frequency within given pstate in (kHz)
			} single;

			struct _range
			{
				NvU32                               minFreq_kHz;		//! Min clock frequency within given pstate in (kHz)
				NvU32                               maxFreq_kHz;		//! Max clock frequency within given pstate in (kHz)
				NV_GPU_PERF_VOLTAGE_INFO_DOMAIN_ID  domainId;			//! Voltage domain ID and value range in (uV) required for this clock
				NvU32                               minVoltage_uV;
				NvU32                               maxVoltage_uV;
			} range;
		} data;
	} NV_GPU_PSTATE20_CLOCK_ENTRY_V1;

	typedef struct _NV_GPU_PSTATE20_BASE_VOLTAGE_ENTRY_V1				//! Used to describe single base voltage entry
	{
		NV_GPU_PERF_VOLTAGE_INFO_DOMAIN_ID			domainId;			//! ID of the voltage domain
		NvU32										bIsEditable : 1;
		NvU32										reserved : 31;		//! These bits are reserved for future use (must be always 0)
		NvU32										volt_uV;			//! Current base voltage settings in [uV]
		NV_GPU_PERF_PSTATES20_PARAM_DELTA			voltDelta_uV;		// Current base voltage delta from nominal settings in [uV]
	} NV_GPU_PSTATE20_BASE_VOLTAGE_ENTRY_V1;

	typedef struct _NV_GPU_PERF_PSTATES20_INFO_V1	//! Used in NvAPI_GPU_GetPstates20() interface call.
	{
		NvU32										version;			//! Version info of the structure (NV_GPU_PERF_PSTATES20_INFO_VER<n>)
		NvU32										bIsEditable : 1;
		NvU32										reserved : 31;		//! These bits are reserved for future use (must be always 0)
		NvU32										numPstates;			//! Number of populated pstates
		NvU32										numClocks;			//! Number of populated clocks (per pstate)
		NvU32										numBaseVoltages;	//! Number of populated base voltages (per pstate)

		//! Performance state (P-State) settings
		//! Valid index range is 0 to numPstates-1
		struct
		{
			NV_GPU_PERF_PSTATE_ID                   pstateId;			//! ID of the P-State
			NvU32                                   bIsEditable : 1;
			NvU32                                   reserved : 31;		//! These bits are reserved for future use (must be always 0)
																		//! Array of clock entries
																		//! Valid index range is 0 to numClocks-1
			NV_GPU_PSTATE20_CLOCK_ENTRY_V1          clocks[NVAPI_MAX_GPU_PSTATE20_CLOCKS];
			//! Array of baseVoltage entries
			//! Valid index range is 0 to numBaseVoltages-1
			NV_GPU_PSTATE20_BASE_VOLTAGE_ENTRY_V1   baseVoltages[NVAPI_MAX_GPU_PSTATE20_BASE_VOLTAGES];
		} pstates[NVAPI_MAX_GPU_PSTATE20_PSTATES];
	} NV_GPU_PERF_PSTATES20_INFO_V1;
#	define NV_GPU_PERF_PSTATES20_INFO_VER_1			MAKE_NVAPI_VERSION(NV_GPU_PERF_PSTATES20_INFO_V1, 1)

	typedef struct _NV_GPU_CLOCK_FREQUENCIES_V2
	{
		NvU32										version;			//!< Structure version
		NvU32										ClockType : 4;		//!< One of NV_GPU_CLOCK_FREQUENCIES_CLOCK_TYPE. Used to specify the type of clock to be returned.
		NvU32										reserved : 20;		//!< These bits are reserved for future use. Must be set to 0.
		NvU32										reserved1 : 8;		//!< These bits are reserved.
		struct
		{
			NvU32									bIsPresent : 1;		//!< Set if this domain is present on this GPU
			NvU32									reserved : 31;		//!< These bits are reserved for future use.
			NvU32									frequency;			//!< Clock frequency (kHz)
		}domain[NVAPI_MAX_GPU_PUBLIC_CLOCKS];
	} NV_GPU_CLOCK_FREQUENCIES_V2;
#	define NV_GPU_CLOCK_FREQUENCIES_VER_2			MAKE_NVAPI_VERSION(NV_GPU_CLOCK_FREQUENCIES_V2, 2)

	typedef struct _NV_GPU_THERMAL_SETTINGS_V2
	{
		NvU32										version;			//!< structure version
		NvU32										count;				//!< number of associated thermal sensors
		struct
		{
			NV_THERMAL_CONTROLLER					controller;			//!< internal, ADM1032, MAX6649...
			NvS32									defaultMinTemp;		//!< Minimum default temperature value of the thermal sensor in degree Celsius
			NvS32									defaultMaxTemp;		//!< Maximum default temperature value of the thermal sensor in degree Celsius
			NvS32									currentTemp;		//!< Current temperature value of the thermal sensor in degree Celsius
			NV_THERMAL_TARGET						target;				//!< Thermal sensor targeted - GPU, memory, chipset, powersupply, Visual Computing Device, etc
		} sensor[NVAPI_MAX_THERMAL_SENSORS_PER_GPU];
	} NV_GPU_THERMAL_SETTINGS_V2;
#	define NV_GPU_THERMAL_SETTINGS_VER_2			MAKE_NVAPI_VERSION(NV_GPU_THERMAL_SETTINGS_V2, 2)

	typedef struct _NV_GPU_DYNAMIC_PSTATES_INFO_EX
	{
		NvU32										version;			//!< Structure version
		NvU32										flags;				//!< bit 0 indicates if the dynamic Pstate is enabled or not
		struct
		{
			NvU32									bIsPresent : 1;		//!< Set if this utilization domain is present on this GPU
			NvU32									percentage;			//!< Percentage of time where the domain is considered busy in the last 1 second interval
		} utilization[NVAPI_MAX_GPU_UTILIZATIONS];
	} NV_GPU_DYNAMIC_PSTATES_INFO_EX;
#	define NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER		MAKE_NVAPI_VERSION(NV_GPU_DYNAMIC_PSTATES_INFO_EX, 1)

	typedef struct _NVAPI_GPU_POWER_STATUS
	{
		NvU32										version;			//!< Structure version
		NvU32										flags;
		struct
		{
			NvU32									unknown1;
			NvU32									unknown2;
			NvU32									power;				// Power limit * 1000
			NvU32									unknown3;
		} entries[4];
	} NVAPI_GPU_POWER_STATUS;
#	define NVAPI_GPU_POWER_STATUS_VER				MAKE_NVAPI_VERSION(NVAPI_GPU_POWER_STATUS, 1)

	typedef struct _NVAPI_GPU_THERMAL_LIMIT_V2
	{
		NvU32										version;			//!< Structure version
		NvU32										flags;
		struct
		{
			NV_THERMAL_CONTROLLER					controller;
			NvS32									value;
			NvU32									flags;
		} entries[4];
	} NVAPI_GPU_THERMAL_LIMIT_V2;
#	define NVAPI_GPU_THERMAL_LIMIT_VER_2			MAKE_NVAPI_VERSION(NVAPI_GPU_THERMAL_LIMIT_V2, 2)

	typedef struct _NV_GPU_COOLER_SETTINGS_V2
	{
		NvU32										version;			//!< Structure version
		NvU32										flags;
		struct
		{
			NvU32									type;
			NvU32									controller;
			NvS32									defaultMin;
			NvS32									defaultMax;
			NvS32									currentMin;
			NvS32									currentMax;
			NvS32									currentLevel;
			NvU32									defaultPolicy;
			NvU32									currentPolicy;
			NvS32									target;
			NvU32									controlType;
			NvU32									active;
		} cooler[NVAPI_MAX_COOLERS_PER_GPU];
	} NV_GPU_COOLER_SETTINGS_V2;
#	define NV_GPU_COOLER_SETTINGS_VER_2				MAKE_NVAPI_VERSION(NV_GPU_COOLER_SETTINGS_V2, 2)

	typedef struct _NV_DISPLAY_DRIVER_MEMORY_INFO_V3
	{
		NvU32										version;
		NvU32										dedicatedVideoMemory;
		NvU32										availableDedicatedVideoMemory;
		NvU32										systemVideoMemory;
		NvU32										sharedSystemMemory;
		NvU32										curAvailableDedicatedVideoMemory;
		NvU32										dedicatedVideoMemoryEvictionsSize;
		NvU32										dedicatedVideoMemoryEvictionCount;
	} NV_DISPLAY_DRIVER_MEMORY_INFO_V3;
#	define NV_DISPLAY_DRIVER_MEMORY_INFO_VER_3		MAKE_NVAPI_VERSION(NV_DISPLAY_DRIVER_MEMORY_INFO_V3, 2)

	typedef void *(*QueryInterface_t)							(NvAPI_FUNCTIONS offset);
	typedef NvAPI_Status(*GetErrorMessage_t)					(NvAPI_Status status, NvAPI_ShortString message);

	typedef NvAPI_Status(*Initialize_t)							();
	typedef NvAPI_Status(*Unload_t)								();
	typedef NvAPI_Status(*EnumPhysicalGPUs_t)					(NvPhysicalGpuHandle gpuHandles[NVAPI_MAX_PHYSICAL_GPUS], NvU32 *gpuCount);
	typedef NvAPI_Status(*GPU_GetBusID_t)						(NvPhysicalGpuHandle handle, NvU32 *busid);

	typedef NvAPI_Status(*GPU_GetAllClockFrequencies_t)			(NvPhysicalGpuHandle handle, NV_GPU_CLOCK_FREQUENCIES_V2 *pClkFreqs);
	typedef NvAPI_Status(*DLL_ClientPowerPoliciesGetStatus_t)	(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_STATUS *powerStatus);
	typedef NvAPI_Status(*DLL_ClientThermalPoliciesGetLimit_t)	(NvPhysicalGpuHandle handle, NVAPI_GPU_THERMAL_LIMIT_V2 *thermalLimit);
	typedef NvAPI_Status(*GPU_GetCoolersSettings_t)				(NvPhysicalGpuHandle handle, NvU32 coolerIndex, NV_GPU_COOLER_SETTINGS_V2 *coolerSettings);

	typedef NvAPI_Status(*GPU_GetMemoryInfo_t)					(NvPhysicalGpuHandle handle, NV_DISPLAY_DRIVER_MEMORY_INFO_V3 *memoryInfo);

	typedef NvAPI_Status(*GPU_GetTachReading_t)					(NvPhysicalGpuHandle handle, NvU32 *value);
	typedef NvAPI_Status(*GPU_GetThermalSettings_t)				(NvPhysicalGpuHandle handle, NvU32 sensorIndex, NV_GPU_THERMAL_SETTINGS_V2 *thermalSettings);
	typedef NvAPI_Status(*GPU_GetPstates20_t)					(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATES20_INFO_V1 *pstatesInfo);
	typedef NvAPI_Status(*GPU_GetCurrentPstate_t)				(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATE_ID *currentPstate);
	typedef NvAPI_Status(*GPU_GetDynamicPstatesInfoEx_t)		(NvPhysicalGpuHandle handle, NV_GPU_DYNAMIC_PSTATES_INFO_EX *dynamicPstatesInfoEx);
	typedef NvAPI_Status(*GPU_GetPerfDecreaseInfo_t)			(NvPhysicalGpuHandle handle, NVAPI_GPU_PERF_DECREASE *perfDescInfo);

	static QueryInterface_t										QueryInterface;
	static GetErrorMessage_t									GetErrorMessage;

	static Initialize_t											Initialize;
	static Unload_t												Unload;
	static EnumPhysicalGPUs_t									EnumPhysicalGPUs;
	static GPU_GetBusID_t										GPU_GetBusID;

	static GPU_GetAllClockFrequencies_t							GPU_GetAllClockFrequencies;
	static DLL_ClientPowerPoliciesGetStatus_t					DLL_ClientPowerPoliciesGetStatus;
	static DLL_ClientThermalPoliciesGetLimit_t					DLL_ClientThermalPoliciesGetLimit;
	static GPU_GetCoolersSettings_t								GPU_GetCoolersSettings;

	static GPU_GetMemoryInfo_t									GPU_GetMemoryInfo;

	static GPU_GetTachReading_t									GPU_GetTachReading;
	static GPU_GetThermalSettings_t								GPU_GetThermalSettings;
	static GPU_GetPstates20_t									GPU_GetPstates20;
	static GPU_GetCurrentPstate_t								GPU_GetCurrentPstate;
	static GPU_GetDynamicPstatesInfoEx_t						GPU_GetDynamicPstatesInfoEx;
	static GPU_GetPerfDecreaseInfo_t							GPU_GetPerfDecreaseInfo;

	static bool													isInitialized;
	static NvPhysicalGpuHandle									gpuHandles[NVAPI_MAX_PHYSICAL_GPUS];
	static NvU32												gpuCount;

	NvPhysicalGpuHandle deviceHandle;

public:
	static bool FoundNvAPI64();

	static void initialize();
	static void unload();

	const int deviceID;
	NvU32 pciBusID;

	NV_API(const int deviceID, NvU32 pciBusID);

	NvAPI_Status getErrorMessage(NvAPI_Status status, std::string *message);

	NvAPI_Status getDeviceMemory(int *memorySize);

	NvAPI_Status getSettingMaxCoreClock(int *maxCoreClock);
	NvAPI_Status getSettingMaxMemoryClock(int *maxMemoryClock);
	NvAPI_Status getSettingPowerLimit(int *powerLimit);
	NvAPI_Status getSettingThermalLimit(int *thermalLimit);
	NvAPI_Status getSettingFanLevelPercent(int *fanLevel);

	NvAPI_Status getCurrentFanTachometerRPM(int *tachometerRPM);
	NvAPI_Status getCurrentTemperature(int *temperature);
	NvAPI_Status getCurrentCoreClock(int *coreClock);
	NvAPI_Status getCurrentMemoryClock(int *memoryClock);
	NvAPI_Status getCurrentUtilizationPercent(int *utilization);
	NvAPI_Status getCurrentPstate(int *pstate);
	NvAPI_Status getCurrentThrottleReasons(const char *reasons);
};
