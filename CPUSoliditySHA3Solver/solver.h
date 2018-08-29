#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "cpuSolver.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

	__declspec(dllexport) void GetLogicalProcessorsCount(uint32_t *processorCount)
	{
		*processorCount = cpuSolver::getLogicalProcessorsCount();
	}

	__declspec(dllexport) void GetNewSolutionTemplate(const char *kingAddress, const char *solutionTemplate)
	{
		auto newTemplate = cpuSolver::getNewSolutionTemplate(kingAddress);
		auto newTemplateStr = newTemplate.c_str();
		std::memcpy((void *)solutionTemplate, newTemplateStr, UINT256_LENGTH * 2 + 2);
	}

	__declspec(dllexport) GetKingAddressCallback SetOnGetKingAddressHandler(GetKingAddressCallback getKingAddressCallback)
	{
		m_getKingAddressCallback = getKingAddressCallback;
		return getKingAddressCallback;
	}

	__declspec(dllexport) GetSolutionTemplateCallback SetOnGetSolutionTemplateHandler(GetSolutionTemplateCallback getSolutionTemplateCallback)
	{
		m_getSolutionTemplateCallback = getSolutionTemplateCallback;
		return getSolutionTemplateCallback;
	}

	__declspec(dllexport) GetWorkPositionCallback SetOnGetWorkPositionHandler(GetWorkPositionCallback getWorkPositionCallback)
	{
		m_getWorkPositionCallback = getWorkPositionCallback;
		return getWorkPositionCallback;
	}

	__declspec(dllexport) ResetWorkPositionCallback SetOnResetWorkPositionHandler(ResetWorkPositionCallback resetWorkPositionCallback)
	{
		m_resetWorkPositionCallback = resetWorkPositionCallback;
		return resetWorkPositionCallback;
	}

	__declspec(dllexport) IncrementWorkPositionCallback SetOnIncrementWorkPositionHandler(IncrementWorkPositionCallback incrementWorkPositionCallback)
	{
		m_incrementWorkPositionCallback = incrementWorkPositionCallback;
		return incrementWorkPositionCallback;
	}

	__declspec(dllexport) MessageCallback SetOnMessageHandler(MessageCallback messageCallback)
	{
		m_messageCallback = messageCallback;
		return messageCallback;
	}

	__declspec(dllexport) SolutionCallback SetOnSolutionHandler(SolutionCallback solutionCallback)
	{
		m_solutionCallback = solutionCallback;
		return solutionCallback;
	}

	__declspec(dllexport) cpuSolver *GetInstance(const char *threads) noexcept
	{		
		try { return new cpuSolver(threads); }
		catch (...) { return NULL; }
	}

	__declspec(dllexport) void DisposeInstance(cpuSolver *instance) noexcept
	{
		try
		{
			((cpuSolver *)instance)->~cpuSolver();
			free((cpuSolver *)instance);
		}
		catch (...) {}
	}

	__declspec(dllexport) void SetSubmitStale(cpuSolver *instance, const bool submitStale)
	{
		((cpuSolver *)instance)->m_SubmitStale = submitStale;
	}

	__declspec(dllexport) void IsMining(cpuSolver *instance, bool *isMining)
	{
		*isMining = ((cpuSolver *)instance)->isMining();
	}

	__declspec(dllexport) void IsPaused(cpuSolver *instance, bool *isPaused)
	{
		*isPaused = ((cpuSolver *)instance)->isPaused();
	}

	__declspec(dllexport) void GetHashRateByThreadID(cpuSolver *instance, const uint32_t threadID, uint64_t *hashRate)
	{
		*hashRate = ((cpuSolver *)instance)->getHashRateByThreadID(threadID);
	}

	__declspec(dllexport) void GetTotalHashRate(cpuSolver *instance, uint64_t *totalHashRate)
	{
		*totalHashRate = ((cpuSolver *)instance)->getTotalHashRate();
	}

	__declspec(dllexport) void UpdatePrefix(cpuSolver *instance, const char *prefix)
	{
		((cpuSolver *)instance)->updatePrefix(prefix);
	}

	__declspec(dllexport) void UpdateTarget(cpuSolver *instance, const char *target)
	{
		((cpuSolver *)instance)->updateTarget(target);
	}

	__declspec(dllexport) void PauseFinding(cpuSolver *instance, const bool pause)
	{
		((cpuSolver *)instance)->pauseFinding(pause);
	}

	__declspec(dllexport) void StartFinding(cpuSolver *instance)
	{
		((cpuSolver *)instance)->startFinding();
	}

	__declspec(dllexport) void StopFinding(cpuSolver *instance)
	{
		((cpuSolver *)instance)->stopFinding();
	}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !__SOLVER__
