#include "solver.h"

namespace CPUSolver
{
	void GetLogicalProcessorsCount(uint32_t *processorCount)
	{
		*processorCount = cpuSolver::getLogicalProcessorsCount();
	}

	void GetNewSolutionTemplate(const char *kingAddress, const char *solutionTemplate)
	{
		auto newTemplate = cpuSolver::getNewSolutionTemplate(kingAddress);
		auto newTemplateStr = newTemplate.c_str();
		std::memcpy((void *)solutionTemplate, newTemplateStr, UINT256_LENGTH * 2 + 2);
	}

	void *SetOnGetKingAddressHandler(void *instance, void *getKingAddressCallback)
	{
		((cpuSolver *)instance)->m_getKingAddressCallback = (GetKingAddressCallback)getKingAddressCallback;
		return getKingAddressCallback;
	}

	void *SetOnGetSolutionTemplateHandler(void *instance, void *getSolutionTemplateCallback)
	{
		((cpuSolver *)instance)->m_getSolutionTemplateCallback = (GetSolutionTemplateCallback)getSolutionTemplateCallback;
		return getSolutionTemplateCallback;
	}

	void *SetOnGetWorkPositionHandler(void *instance, void *getWorkPositionCallback)
	{
		((cpuSolver *)instance)->m_getWorkPositionCallback = (GetWorkPositionCallback)getWorkPositionCallback;
		return getWorkPositionCallback;
	}

	void *SetOnResetWorkPositionHandler(void *instance, void *resetWorkPositionCallback)
	{
		((cpuSolver *)instance)->m_resetWorkPositionCallback = (ResetWorkPositionCallback)resetWorkPositionCallback;
		return resetWorkPositionCallback;
	}

	void *SetOnIncrementWorkPositionHandler(void *instance, void *incrementWorkPositionCallback)
	{
		((cpuSolver *)instance)->m_incrementWorkPositionCallback = (IncrementWorkPositionCallback)incrementWorkPositionCallback;
		return incrementWorkPositionCallback;
	}

	void *SetOnMessageHandler(void *instance, void *messageCallback)
	{
		((cpuSolver *)instance)->m_messageCallback = (MessageCallback)messageCallback;
		return messageCallback;
	}

	void *SetOnSolutionHandler(void *instance, void *solutionCallback)
	{
		((cpuSolver *)instance)->m_solutionCallback = (SolutionCallback)solutionCallback;
		return solutionCallback;
	}

	void *GetInstance(const char *threads) noexcept
	{
		try { return new cpuSolver(threads); }
		catch (...) { return nullptr; }
	}

	void DisposeInstance(void *instance) noexcept
	{
		try
		{
			((cpuSolver *)instance)->~cpuSolver();
			free(instance);
		}
		catch (...) {}
	}

	void SetSubmitStale(void *instance, const bool submitStale)
	{
		((cpuSolver *)instance)->m_SubmitStale = submitStale;
	}

	void IsMining(void *instance, bool *isMining)
	{
		*isMining = ((cpuSolver *)instance)->isMining();
	}

	void IsPaused(void *instance, bool *isPaused)
	{
		*isPaused = ((cpuSolver *)instance)->isPaused();
	}

	void GetHashRateByThreadID(void *instance, const uint32_t threadID, uint64_t *hashRate)
	{
		*hashRate = ((cpuSolver *)instance)->getHashRateByThreadID(threadID);
	}

	void GetTotalHashRate(void *instance, uint64_t *totalHashRate)
	{
		*totalHashRate = ((cpuSolver *)instance)->getTotalHashRate();
	}

	void UpdatePrefix(void *instance, const char *prefix)
	{
		((cpuSolver *)instance)->updatePrefix(prefix);
	}

	void UpdateTarget(void *instance, const char *target)
	{
		((cpuSolver *)instance)->updateTarget(target);
	}

	void PauseFinding(void *instance, const bool pause)
	{
		((cpuSolver *)instance)->pauseFinding(pause);
	}

	void StartFinding(void *instance)
	{
		((cpuSolver *)instance)->startFinding();
	}

	void StopFinding(void *instance)
	{
		((cpuSolver *)instance)->stopFinding();
	}
}