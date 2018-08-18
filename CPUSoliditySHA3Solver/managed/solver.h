#pragma once
#include "ManagedObject.h"
#include "../cpuSolver.h"

namespace CPUSolver
{
	public ref class Solver : public ManagedObject<cpuSolver>
	{
	public:
		delegate void OnGetKingAddressDelegate(uint8_t *);
		delegate void OnGetWorkPositionDelegate(unsigned __int64 %);
		delegate void OnResetWorkPositionDelegate(unsigned __int64 %);
		delegate void OnIncrementWorkPositionDelegate(unsigned __int64 %, unsigned __int64);
		delegate void OnGetSolutionTemplateDelegate(uint8_t *);
		delegate void OnMessageDelegate(int, System::String ^, System::String ^);
		delegate void OnSolutionDelegate(System::String ^, System::String ^, System::String ^, System::String ^, System::String ^, System::String ^, bool);

		OnGetKingAddressDelegate ^OnGetKingAddressHandler;
		OnGetWorkPositionDelegate ^OnGetWorkPositionHandler;
		OnResetWorkPositionDelegate ^OnResetWorkPositionHandler;
		OnIncrementWorkPositionDelegate ^OnIncrementWorkPositionHandler;
		OnGetSolutionTemplateDelegate ^OnGetSolutionTemplateHandler;
		OnMessageDelegate ^ OnMessageHandler;
		OnSolutionDelegate ^ OnSolutionHandler;

	private:
		OnGetKingAddressDelegate ^ m_managedOnGetKingAddress;
		OnGetWorkPositionDelegate ^m_managedOnGetWorkPosition;
		OnResetWorkPositionDelegate ^m_managedOnResetWorkPosition;
		OnIncrementWorkPositionDelegate ^m_managedOnIncrementWorkPosition;
		OnGetSolutionTemplateDelegate ^m_managedOnGetSolutionTemplate;
		OnMessageDelegate ^ m_managedOnMessage;
		OnSolutionDelegate ^ m_managedOnSolution;

	public:
		static unsigned int getLogicalProcessorsCount();
		static System::String ^getNewSolutionTemplate(System::String ^kingAddress);

	public:
		// require web3 contract getMethod -> _MAXIMUM_TARGET
		Solver(System::String ^maxDifficulty, System::String ^threads);
		~Solver();

		void setCustomDifficulty(uint32_t customDifficulty);
		void setSubmitStale(bool submitStale);
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
		// individual hashrate by threadID, in H/s
		uint64_t getHashRateByThreadID(unsigned int const threadID);

	private:
		void OnGetKingAddress(uint8_t *kingAddress);
		void OnGetWorkPosition(unsigned __int64 %workPosition);
		void OnResetWorkPosition(unsigned __int64 %lastPosition);
		void OnIncrementWorkPosition(unsigned __int64 %lastPosition, unsigned __int64 increment);
		void OnGetSolutionTemplate(uint8_t *solutionTemplate);
		void OnMessage(int threadID, System::String^ type, System::String^ message);
		void OnSolution(System::String^ digest, System::String^ address, System::String^ challenge, System::String^ difficulty, System::String^ target, System::String^ solution, bool isCustomDifficulty);
	};
}
