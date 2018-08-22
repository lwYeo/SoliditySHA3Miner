#pragma once
#include "ManagedObject.h"
#include "../cudaSolver.h"

namespace CudaSolver
{
	public ref class Solver : public ManagedObject<CUDASolver>
	{
	public:
		delegate void OnGetKingAddressDelegate(uint8_t *);
		delegate void OnGetSolutionTemplateDelegate(uint8_t *);
		delegate void OnGetWorkPositionDelegate(unsigned __int64 %);
		delegate void OnResetWorkPositionDelegate(unsigned __int64 %);
		delegate void OnIncrementWorkPositionDelegate(unsigned __int64 %, unsigned __int64);
		delegate void OnMessageDelegate(int, System::String ^, System::String ^);
		delegate void OnSolutionDelegate(System::String ^, System::String ^, System::String ^, System::String ^, System::String ^, System::String ^, bool);

		OnGetKingAddressDelegate ^OnGetKingAddressHandler;
		OnGetSolutionTemplateDelegate ^OnGetSolutionTemplateHandler;
		OnGetWorkPositionDelegate ^OnGetWorkPositionHandler;
		OnResetWorkPositionDelegate ^OnResetWorkPositionHandler;
		OnIncrementWorkPositionDelegate ^OnIncrementWorkPositionHandler;
		OnMessageDelegate ^OnMessageHandler;
		OnSolutionDelegate ^OnSolutionHandler;

	private:
		OnGetKingAddressDelegate ^m_managedOnGetKingAddress;
		OnGetSolutionTemplateDelegate ^m_managedOnGetSolutionTemplate;
		OnGetWorkPositionDelegate ^m_managedOnGetWorkPosition;
		OnResetWorkPositionDelegate ^m_managedOnResetWorkPosition;
		OnIncrementWorkPositionDelegate ^m_managedOnIncrementWorkPosition;
		OnMessageDelegate ^m_managedOnMessage;
		OnSolutionDelegate ^m_managedOnSolution;

	public:
		static bool foundNvAPI64();
		static int getDeviceCount(System::String ^%errorMessage);
		static System::String ^getDeviceName(int deviceID, System::String ^%errorMessage);

	public:
		// require web3 contract getMethod -> _MAXIMUM_TARGET
		Solver(System::String ^maxDifficulty);
		~Solver();

		void setCustomDifficulty(uint32_t customDifficulty);
		void setSubmitStale(bool submitStale);
		bool assignDevice(int const deviceID, float const intensity);
		bool isAssigned();
		bool isAnyInitialised();
		bool isMining();
		bool isPaused();

		// must be in challenge (byte32) + address (byte20) hexadecimal format with "0x" prefix
		void updatePrefix(System::String ^prefix);
		// must be in byte32 hexadecimal format with "0x" prefix
		void updateTarget(System::String ^target);
		// can be in either numeric or hexadecimal format
		void updateDifficulty(System::String ^difficulty);

		void startFinding();
		void stopFinding();
		void pauseFinding(bool pauseFinding);

		// combined hashrate, in H/s
		uint64_t getTotalHashRate();
		// individual hashrate by deviceID, in H/s
		uint64_t getHashRateByDeviceID(int const deviceID);

		int getDeviceSettingMaxCoreClock(int deviceID);
		int getDeviceSettingMaxMemoryClock(int deviceID);
		int getDeviceSettingPowerLimit(int deviceID);
		int getDeviceSettingThermalLimit(int deviceID);
		int getDeviceSettingFanLevelPercent(int deviceID);

		int getDeviceCurrentFanTachometerRPM(int deviceID);
		int getDeviceCurrentTemperature(int deviceID);
		int getDeviceCurrentCoreClock(int deviceID);
		int getDeviceCurrentMemoryClock(int deviceID);
		int getDeviceCurrentUtilizationPercent(int deviceID);
		int getDeviceCurrentPstate(int deviceID);
		System::String ^getDeviceCurrentThrottleReasons(int deviceID);

	private:
		void OnGetKingAddress(uint8_t *kingAddress);
		void OnGetSolutionTemplate(uint8_t *solutionTemplate);
		void OnGetWorkPosition(unsigned __int64 %workPosition);
		void OnResetWorkPosition(unsigned __int64 %lastPosition);
		void OnIncrementWorkPosition(unsigned __int64 %lastPosition, unsigned __int64 increment);
		void OnMessage(int deviceID, System::String ^type, System::String ^message);
		void OnSolution(System::String ^digest, System::String ^address, System::String ^challenge, System::String ^difficulty, System::String ^target, System::String ^solution, bool isCustomDifficulty);
	};
}
