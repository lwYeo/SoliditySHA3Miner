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

	GetKingAddressCallback SetOnGetKingAddressHandler(cpuSolver *instance, GetKingAddressCallback getKingAddressCallback)
	{
		instance->m_getKingAddressCallback = getKingAddressCallback;
		return getKingAddressCallback;
	}

	GetSolutionTemplateCallback SetOnGetSolutionTemplateHandler(cpuSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback)
	{
		instance->m_getSolutionTemplateCallback = getSolutionTemplateCallback;
		return getSolutionTemplateCallback;
	}

	GetWorkPositionCallback SetOnGetWorkPositionHandler(cpuSolver *instance, GetWorkPositionCallback getWorkPositionCallback)
	{
		instance->m_getWorkPositionCallback = getWorkPositionCallback;
		return getWorkPositionCallback;
	}

	ResetWorkPositionCallback SetOnResetWorkPositionHandler(cpuSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback)
	{
		instance->m_resetWorkPositionCallback = resetWorkPositionCallback;
		return resetWorkPositionCallback;
	}

	IncrementWorkPositionCallback SetOnIncrementWorkPositionHandler(cpuSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback)
	{
		instance->m_incrementWorkPositionCallback = incrementWorkPositionCallback;
		return incrementWorkPositionCallback;
	}

	MessageCallback SetOnMessageHandler(cpuSolver *instance, MessageCallback messageCallback)
	{
		instance->m_messageCallback = messageCallback;
		return messageCallback;
	}

	SolutionCallback SetOnSolutionHandler(cpuSolver *instance, SolutionCallback solutionCallback)
	{
		instance->m_solutionCallback = solutionCallback;
		return solutionCallback;
	}

	cpuSolver *GetInstance(const char *threads) noexcept
	{
		try { return new cpuSolver(threads); }
		catch (...) { return nullptr; }
	}

	void DisposeInstance(cpuSolver *instance) noexcept
	{
		try
		{
			instance->~cpuSolver();
			free(instance);
		}
		catch (...) {}
	}

	void SetSubmitStale(cpuSolver *instance, const bool submitStale)
	{
		instance->m_SubmitStale = submitStale;
	}

	void IsMining(cpuSolver *instance, bool *isMining)
	{
		*isMining = instance->isMining();
	}

	void IsPaused(cpuSolver *instance, bool *isPaused)
	{
		*isPaused = instance->isPaused();
	}

	void GetHashRateByThreadID(cpuSolver *instance, const uint32_t threadID, uint64_t *hashRate)
	{
		*hashRate = instance->getHashRateByThreadID(threadID);
	}

	void GetTotalHashRate(cpuSolver *instance, uint64_t *totalHashRate)
	{
		*totalHashRate = instance->getTotalHashRate();
	}

	void UpdatePrefix(cpuSolver *instance, const char *prefix)
	{
		instance->updatePrefix(prefix);
	}

	void UpdateTarget(cpuSolver *instance, const char *target)
	{
		instance->updateTarget(target);
	}

	void PauseFinding(cpuSolver *instance, const bool pause)
	{
		instance->pauseFinding(pause);
	}

	void StartFinding(cpuSolver *instance)
	{
		instance->startFinding();
	}

	void StopFinding(cpuSolver *instance)
	{
		instance->stopFinding();
	}
}