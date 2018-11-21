/*
   Copyright 2018 Lip Wee Yeo Amano

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#ifndef __SOLVER__
#define __SOLVER__

#include "cpuSolver.h"
#include "instance.h"

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
		EXPORT void __CDECL__ SHA3(byte32_t *message, byte32_t *digest);

		EXPORT void __CDECL__ GetCpuName(const char *cpuName);

		EXPORT CpuSolver *__CDECL__ GetInstance() noexcept;

		EXPORT void __CDECL__ DisposeInstance(CpuSolver *instance) noexcept;

		EXPORT void __CDECL__ SetThreadAffinity(CpuSolver *instance, int affinityMask, const char *errorMessage);

		EXPORT void __CDECL__ HashMessage(CpuSolver *instance, Instance *deviceInstance, Processor *processor);

		EXPORT void __CDECL__ HashMidState(CpuSolver *instance, Instance *deviceInstance, Processor *processor);
	}
}

#endif // !__SOLVER__