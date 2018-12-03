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

#include "types.h"

struct Processor
{
	int Affinity;
	uint64_t WorkSize;
	uint64_t WorkPosition;

	uint32_t MaxSolutionCount;
	uint32_t SolutionCount;
	uint64_t *Solutions;
};

struct Instance
{
	int ProcessorCount;
	Processor *Processors;

	uint8_t *SolutionTemplate;
	uint8_t *Message;
	uint8_t *MidState;

	uint8_t *Target;
	uint64_t *High64Target;
};