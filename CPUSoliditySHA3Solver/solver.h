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

		EXPORT void *CDECL GetInstance(const char *threads) noexcept;

		EXPORT void CDECL DisposeInstance(void *instance) noexcept;

		EXPORT void *CDECL SetOnGetKingAddressHandler(void *instance, void *getKingAddressCallback);

		EXPORT void *CDECL SetOnGetSolutionTemplateHandler(void *instance, void *getSolutionTemplateCallback);

		EXPORT void *CDECL SetOnGetWorkPositionHandler(void *instance, void *getWorkPositionCallback);

		EXPORT void *CDECL SetOnResetWorkPositionHandler(void *instance, void *resetWorkPositionCallback);

		EXPORT void *CDECL SetOnIncrementWorkPositionHandler(void *instance, void *incrementWorkPositionCallback);

		EXPORT void *CDECL SetOnMessageHandler(void *instance, void *messageCallback);

		EXPORT void *CDECL SetOnSolutionHandler(void *instance, void *solutionCallback);

		EXPORT void CDECL SetSubmitStale(void *instance, const bool submitStale);

		EXPORT void CDECL IsMining(void *instance, bool *isMining);

		EXPORT void CDECL IsPaused(void *instance, bool *isPaused);

		EXPORT void CDECL GetHashRateByThreadID(void *instance, const uint32_t threadID, uint64_t *hashRate);

		EXPORT void CDECL GetTotalHashRate(void *instance, uint64_t *totalHashRate);

		EXPORT void CDECL UpdatePrefix(void *instance, const char *prefix);

		EXPORT void CDECL UpdateTarget(void *instance, const char *target);

		EXPORT void CDECL PauseFinding(void *instance, const bool pause);

		EXPORT void CDECL StartFinding(void *instance);

		EXPORT void CDECL StopFinding(void *instance);
	}
}

#endif // !__SOLVER__
