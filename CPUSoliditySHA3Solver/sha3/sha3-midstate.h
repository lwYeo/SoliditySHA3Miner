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

#ifndef SHA3_MIDSTATE
#define SHA3_MIDSTATE

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void sha3_midstate(uint64_t const *midState, uint64_t const target, uint64_t const workPosition,
					uint32_t const maxSolutionCount, uint32_t *solutionCount, uint64_t *solutions);

#ifdef __cplusplus
}
#endif

#endif // !SHA3_MIDSTATE