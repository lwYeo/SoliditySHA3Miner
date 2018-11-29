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

#include "cpuSolver.h"

namespace CPUSolver
{
	void CpuSolver::SHA3(byte32_t *message, byte32_t *digest)
	{
		keccak_256(&(*digest)[0], UINT256_LENGTH, &(*message)[0], MESSAGE_LENGTH);
	}

	void CpuSolver::GetCpuName(const char *cpuName)
	{
#	if defined(_MSC_VER)
		int info[4];
		__cpuidex(info, 0x80000000, 0);

		if (info[0] < 0x80000004) return;

		int x = 0;
		for (uint32_t i{ 0x80000002 }; i < 0x80000005; ++i, x += 16)
		{
			__cpuidex(info, i, 0);
			std::memcpy((void *)&cpuName[x], (void *)info, 16);
		}
		std::memset((void *)&cpuName[x + 1], 0, 1);
#	endif
	}

#ifdef __linux__

#include <sched.h>

	void CpuSolver::SetThreadAffinity(int affinityMask, const char *errorMessage)
	{
		cpu_set_t mask_set{ 0 };
		CPU_SET(affinityMask, &mask_set);

		if (sched_setaffinity(0, sizeof(cpu_set_t), &mask_set) != 0)
		{
			auto errMessage = "Failed to set processor affinity (" + std::to_string(affinityMask) + ")";
			auto errMessageChar = errMessage.c_str();
			std::memcpy((void *)errorMessage, errMessageChar, errMessage.length());
			std::memset((void *)&errorMessage[errMessage.length()], 0, 1);
		}
	}

#else

#include <Windows.h>

	void CpuSolver::SetThreadAffinity(int affinityMask, const char *errorMessage)
	{
		if (!SetThreadAffinityMask(GetCurrentThread(), 1ull << affinityMask))
		{
			auto errMessage = "Failed to set processor affinity (" + std::to_string(affinityMask) + ")";
			auto errMessageChar = errMessage.c_str();
			std::memcpy((void *)errorMessage, errMessageChar, errMessage.length());
			std::memset((void *)&errorMessage[errMessage.length()], 0, 1);
		}
	}

#endif

	void CpuSolver::HashMessage(Instance *deviceInstance, Processor *processor)
	{
		byte32_t digest;
		byte32_t currentTarget;
		byte32_t currentSolution;
		message_ut currentMessage;

		std::memcpy(&currentMessage, deviceInstance->Message, MESSAGE_LENGTH);
		std::memcpy(&currentTarget, deviceInstance->Target, UINT256_LENGTH);
		std::memcpy(&currentSolution, deviceInstance->SolutionTemplate, UINT256_LENGTH);

		uint64_t const endWorkPosition = processor->WorkPosition + processor->WorkSize;
		uint32_t const maxSolutionCount = processor->MaxSolutionCount;

		for (auto currentWorkPosition = processor->WorkPosition; currentWorkPosition < endWorkPosition; ++currentWorkPosition)
		{
			std::memcpy(&currentSolution[ADDRESS_LENGTH], &currentWorkPosition, UINT64_LENGTH);
			currentMessage.structure.solution = currentSolution;

			keccak_256(&(digest)[0], UINT256_LENGTH, &currentMessage.byteArray[0], MESSAGE_LENGTH);

			if (IslessThan(digest, currentTarget))
			{
				if (processor->SolutionCount < maxSolutionCount)
				{
					processor->Solutions[processor->SolutionCount] = currentWorkPosition;
					processor->SolutionCount++;
				}
			}
		}
	}

	void CpuSolver::HashMidState(Instance *deviceInstance, Processor *processor)
	{
		uint64_t const endWorkPosition = processor->WorkPosition + processor->WorkSize;
		uint64_t const currentHigh64Target = *deviceInstance->High64Target;
		uint32_t const maxSolutionCount = processor->MaxSolutionCount;

		uint64_t currentMidState[SPONGE_LENGTH / UINT64_LENGTH];
		std::memcpy(&currentMidState, deviceInstance->MidState, SPONGE_LENGTH);

		for (auto currentWorkPosition = processor->WorkPosition; currentWorkPosition < endWorkPosition; ++currentWorkPosition)
		{
			sha3_midstate(currentMidState, currentHigh64Target, currentWorkPosition, maxSolutionCount,
				&processor->SolutionCount, processor->Solutions);
		}
	}

	bool CpuSolver::IslessThan(byte32_t const &left, byte32_t const &right)
	{
		for (uint32_t i{ 0 }; i < UINT256_LENGTH; ++i)
		{
			if (left[i] < right[i]) return true;
			else if (left[i] > right[i]) return false;
		}
		return false;
	}
}