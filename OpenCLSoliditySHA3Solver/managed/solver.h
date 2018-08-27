#pragma once
#include "managedObject.h"
#include "../openCLSolver.h"

namespace OpenCLSolver
{
	public ref class Solver : public ManagedObject<openCLSolver>
	{
	public:
		delegate void OnGetKingAddressDelegate(uint8_t *);
		delegate void OnGetSolutionTemplateDelegate(uint8_t *);
		delegate void OnGetWorkPositionDelegate(unsigned __int64 %);
		delegate void OnResetWorkPositionDelegate(unsigned __int64 %);
		delegate void OnIncrementWorkPositionDelegate(unsigned __int64 %, unsigned __int64);
		delegate void OnMessageDelegate(System::String ^, int, System::String ^, System::String ^);
		delegate void OnSolutionDelegate(System::String ^, System::String ^, System::String ^, System::String ^, System::String ^);

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
		static void preInitialize(bool allowIntel, System::String ^%errorMessage);
		static bool foundAdlApi();
		static System::String ^getPlatformNames();
		static int getDeviceCount(System::String ^platformName, System::String ^%errorMessage);
		static System::String ^getDeviceName(System::String ^platformName, int deviceID, System::String ^%errorMessage);

	public:
		// require web3 contract getMethod -> _MAXIMUM_TARGET
		Solver();
		~Solver();

		void setSubmitStale(bool submitStale);
		bool assignDevice(System::String ^platformName, int const deviceID, float %intensity);
		bool isAssigned();
		bool isAnyInitialised();
		bool isMining();
		bool isPaused();

		// must be in challenge (byte32) + address (byte20) hexadecimal format with "0x" prefix
		void updatePrefix(System::String ^prefix);
		// must be in byte32 hexadecimal format with "0x" prefix
		void updateTarget(System::String ^target);

		void startFinding();
		void stopFinding();
		void pauseFinding(bool pauseFinding);

		// combined hashrate, in H/s
		uint64_t getTotalHashRate();
		// individual hashrate by deviceID, in H/s
		uint64_t getHashRateByDevice(System::String ^platformName, int const deviceID);

		System::String ^getDeviceName(System::String ^platformName, int const deviceID);

		int getDeviceSettingMaxCoreClock(System::String ^platformName, int const deviceID);
		int getDeviceSettingMaxMemoryClock(System::String ^platformName, int const deviceID);
		int getDeviceSettingPowerLimit(System::String ^platformName, int const deviceID);
		int getDeviceSettingThermalLimit(System::String ^platformName, int const deviceID);
		int getDeviceSettingFanLevelPercent(System::String ^platformName, int const deviceID);

		int getDeviceCurrentFanTachometerRPM(System::String ^platformName, int const deviceID);
		int getDeviceCurrentTemperature(System::String ^platformName, int const deviceID);
		int getDeviceCurrentCoreClock(System::String ^platformName, int const deviceID);
		int getDeviceCurrentMemoryClock(System::String ^platformName, int const deviceID);
		int getDeviceCurrentUtilizationPercent(System::String ^platformName, int const deviceID);

	private:
		void OnGetKingAddress(uint8_t *kingAddress);
		void OnGetSolutionTemplate(uint8_t *solutionTemplate);
		void OnGetWorkPosition(unsigned __int64 %workPosition);
		void OnResetWorkPosition(unsigned __int64 %lastPosition);
		void OnIncrementWorkPosition(unsigned __int64 %lastPosition, unsigned __int64 increment);
		void OnMessage(System::String ^platformName, int deviceID, System::String ^type, System::String ^message);
		void OnSolution(System::String ^digest, System::String ^address, System::String ^challenge, System::String ^target, System::String ^solution);
	};
}
