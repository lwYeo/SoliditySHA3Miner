#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "cpuSolver.h"

#ifdef __linux__
#	define EXPORT
#	define CDECL
#else
#	define EXPORT _declspec(dllexport)
#	define CDECL __cdecl
#endif

namespace CPUSolver
{
	extern "C"
	{
		EXPORT void CDECL GetLogicalProcessorsCount(uint32_t *processorCount);

		EXPORT void CDECL GetNewSolutionTemplate(const char *kingAddress, const char *solutionTemplate);

		EXPORT cpuSolver *CDECL GetInstance(const char *threads) noexcept;

		EXPORT void CDECL DisposeInstance(cpuSolver *instance) noexcept;

		EXPORT GetKingAddressCallback CDECL SetOnGetKingAddressHandler(cpuSolver *instance, GetKingAddressCallback getKingAddressCallback);

		EXPORT GetSolutionTemplateCallback CDECL SetOnGetSolutionTemplateHandler(cpuSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		EXPORT GetWorkPositionCallback CDECL SetOnGetWorkPositionHandler(cpuSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		EXPORT ResetWorkPositionCallback CDECL SetOnResetWorkPositionHandler(cpuSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		EXPORT IncrementWorkPositionCallback CDECL SetOnIncrementWorkPositionHandler(cpuSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		EXPORT MessageCallback CDECL SetOnMessageHandler(cpuSolver *instance, MessageCallback messageCallback);

		EXPORT SolutionCallback CDECL SetOnSolutionHandler(cpuSolver *instance, SolutionCallback solutionCallback);

		EXPORT void CDECL SetSubmitStale(cpuSolver *instance, const bool submitStale);

		EXPORT void CDECL IsMining(cpuSolver *instance, bool *isMining);

		EXPORT void CDECL IsPaused(cpuSolver *instance, bool *isPaused);

		EXPORT void CDECL GetHashRateByThreadID(cpuSolver *instance, const uint32_t threadID, uint64_t *hashRate);

		EXPORT void CDECL GetTotalHashRate(cpuSolver *instance, uint64_t *totalHashRate);

		EXPORT void CDECL UpdatePrefix(cpuSolver *instance, const char *prefix);

		EXPORT void CDECL UpdateTarget(cpuSolver *instance, const char *target);

		EXPORT void CDECL PauseFinding(cpuSolver *instance, const bool pause);

		EXPORT void CDECL StartFinding(cpuSolver *instance);

		EXPORT void CDECL StopFinding(cpuSolver *instance);
	
	}
}

#endif // !__SOLVER__
