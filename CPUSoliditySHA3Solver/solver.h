#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "cpuSolver.h"

#ifdef __linux__
#	define EXPORT
#	define __CDECL__
#else
#	define EXPORT _declspec(dllexport)
#	define __CDECL__ __cdecl
#endif

namespace CPUSolver
{
	extern "C"
	{
		EXPORT void __CDECL__ GetLogicalProcessorsCount(uint32_t *processorCount);

		EXPORT void __CDECL__ GetNewSolutionTemplate(const char *kingAddress, const char *solutionTemplate);

		EXPORT cpuSolver *__CDECL__ GetInstance(const char *threads) noexcept;

		EXPORT void __CDECL__ DisposeInstance(cpuSolver *instance) noexcept;

		EXPORT GetKingAddressCallback __CDECL__ SetOnGetKingAddressHandler(cpuSolver *instance, GetKingAddressCallback getKingAddressCallback);

		EXPORT GetSolutionTemplateCallback __CDECL__ SetOnGetSolutionTemplateHandler(cpuSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		EXPORT GetWorkPositionCallback __CDECL__ SetOnGetWorkPositionHandler(cpuSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		EXPORT ResetWorkPositionCallback __CDECL__ SetOnResetWorkPositionHandler(cpuSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		EXPORT IncrementWorkPositionCallback __CDECL__ SetOnIncrementWorkPositionHandler(cpuSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		EXPORT MessageCallback __CDECL__ SetOnMessageHandler(cpuSolver *instance, MessageCallback messageCallback);

		EXPORT SolutionCallback __CDECL__ SetOnSolutionHandler(cpuSolver *instance, SolutionCallback solutionCallback);

		EXPORT void __CDECL__ SetSubmitStale(cpuSolver *instance, const bool submitStale);

		EXPORT void __CDECL__ IsMining(cpuSolver *instance, bool *isMining);

		EXPORT void __CDECL__ IsPaused(cpuSolver *instance, bool *isPaused);

		EXPORT void __CDECL__ GetHashRateByThreadID(cpuSolver *instance, const uint32_t threadID, uint64_t *hashRate);

		EXPORT void __CDECL__ GetTotalHashRate(cpuSolver *instance, uint64_t *totalHashRate);

		EXPORT void __CDECL__ UpdatePrefix(cpuSolver *instance, const char *prefix);

		EXPORT void __CDECL__ UpdateTarget(cpuSolver *instance, const char *target);

		EXPORT void __CDECL__ PauseFinding(cpuSolver *instance, const bool pause);

		EXPORT void __CDECL__ StartFinding(cpuSolver *instance);

		EXPORT void __CDECL__ StopFinding(cpuSolver *instance);
	
	}
}

#endif // !__SOLVER__
