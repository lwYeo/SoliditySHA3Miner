#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "cpuSolver.h"

namespace CPUSolver
{
	extern "C"
	{
		__declspec(dllexport) void __cdecl GetLogicalProcessorsCount(uint32_t *processorCount);

		__declspec(dllexport) void __cdecl GetNewSolutionTemplate(const char *kingAddress, const char *solutionTemplate);

		__declspec(dllexport) cpuSolver *__cdecl GetInstance(const char *threads) noexcept;

		__declspec(dllexport) void __cdecl DisposeInstance(cpuSolver *instance) noexcept;

		__declspec(dllexport) GetKingAddressCallback __cdecl SetOnGetKingAddressHandler(cpuSolver *instance, GetKingAddressCallback getKingAddressCallback);

		__declspec(dllexport) GetSolutionTemplateCallback __cdecl SetOnGetSolutionTemplateHandler(cpuSolver *instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

		__declspec(dllexport) GetWorkPositionCallback __cdecl SetOnGetWorkPositionHandler(cpuSolver *instance, GetWorkPositionCallback getWorkPositionCallback);

		__declspec(dllexport) ResetWorkPositionCallback __cdecl SetOnResetWorkPositionHandler(cpuSolver *instance, ResetWorkPositionCallback resetWorkPositionCallback);

		__declspec(dllexport) IncrementWorkPositionCallback __cdecl SetOnIncrementWorkPositionHandler(cpuSolver *instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

		__declspec(dllexport) MessageCallback __cdecl SetOnMessageHandler(cpuSolver *instance, MessageCallback messageCallback);

		__declspec(dllexport) SolutionCallback __cdecl SetOnSolutionHandler(cpuSolver *instance, SolutionCallback solutionCallback);

		__declspec(dllexport) void __cdecl SetSubmitStale(cpuSolver *instance, const bool submitStale);

		__declspec(dllexport) void __cdecl IsMining(cpuSolver *instance, bool *isMining);

		__declspec(dllexport) void __cdecl IsPaused(cpuSolver *instance, bool *isPaused);

		__declspec(dllexport) void __cdecl GetHashRateByThreadID(cpuSolver *instance, const uint32_t threadID, uint64_t *hashRate);

		__declspec(dllexport) void __cdecl GetTotalHashRate(cpuSolver *instance, uint64_t *totalHashRate);

		__declspec(dllexport) void __cdecl UpdatePrefix(cpuSolver *instance, const char *prefix);

		__declspec(dllexport) void __cdecl UpdateTarget(cpuSolver *instance, const char *target);

		__declspec(dllexport) void __cdecl PauseFinding(cpuSolver *instance, const bool pause);

		__declspec(dllexport) void __cdecl StartFinding(cpuSolver *instance);

		__declspec(dllexport) void __cdecl StopFinding(cpuSolver *instance);
	}
}

#endif // !__SOLVER__
