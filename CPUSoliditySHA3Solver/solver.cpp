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

#include "solver.h"

namespace CPUSolver
{
	void SHA3(byte32_t *message, byte32_t *digest)
	{
		CpuSolver::SHA3(message, digest);
	}

	void GetCpuName(const char *cpuName)
	{
		CpuSolver::GetCpuName(cpuName);
	}

	CpuSolver *GetInstance() noexcept
	{
		return new CpuSolver();
	}

	void DisposeInstance(CpuSolver *instance) noexcept
	{
		delete instance;
	}

	void SetThreadAffinity(CpuSolver *instance, int affinityMask, const char *errorMessage)
	{
		instance->SetThreadAffinity(affinityMask, errorMessage);
	}

	void HashMessage(CpuSolver *instance, Instance *deviceInstance, Processor *processor)
	{
		instance->HashMessage(deviceInstance, processor);
	}

	void HashMidState(CpuSolver *instance, Instance *deviceInstance, Processor *processor)
	{
		instance->HashMidState(deviceInstance, processor);
	}
}