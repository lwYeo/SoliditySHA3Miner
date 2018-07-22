#pragma once
#include "ManagedObject.h"
#include "../cudaSolver.h"

namespace CudaSolver
{
	public delegate void OnMessageDelegate(int, System::String^, System::String^);
	public delegate void OnSolutionDelegate(System::String^, System::String^, System::String^, System::String^, System::String^, System::String^, bool);

	public ref class Solver : public ManagedObject<CUDASolver>
	{
	public:
		OnMessageDelegate ^ OnMessageHandler;
		OnSolutionDelegate ^ OnSolutionHandler;

	private:
		OnMessageDelegate ^ m_managedOnMessage;
		OnSolutionDelegate ^ m_managedOnSolution;

	public:
		static int getDeviceCount(System::String^% errorMessage);
		static System::String^ getDeviceName(int deviceID, System::String^% errorMessage);

	public:
		// require web3 contract getMethod -> _MAXIMUM_TARGET
		Solver(System::String^ maxDifficulty, System::String ^solutionTemplate, System::String ^kingAddress);
		~Solver();

		void setCustomDifficulty(uint32_t customDifficulty);
		void setSubmitStale(bool submitStale);
		bool assignDevice(int const deviceID, float const intensity);
		bool isAssigned();
		bool isAnyInitialised();
		bool isMining();
		bool isPaused();

		// must be in challenge (byte32) + address (byte20) hexadecimal format with "0x" prefix
		void updatePrefix(System::String^ prefix);
		// must be in byte32 hexadecimal format with "0x" prefix
		void updateTarget(System::String^ target);
		// can be in either numeric or hexadecimal format
		void updateDifficulty(System::String^ difficulty);

		void startFinding();
		void stopFinding();
		void pauseFinding(bool pauseFinding);

		// combined hashrate, in H/s
		uint64_t getTotalHashRate();
		// individual hashrate by deviceID, in H/s
		uint64_t getHashRateByDeviceID(int const deviceID);

	private:
		void OnMessage(int deviceID, System::String^ type, System::String^ message);
		void OnSolution(System::String^ digest, System::String^ address, System::String^ challenge, System::String^ difficulty, System::String^ target, System::String^ solution, bool isCustomDifficulty);
	};
}
