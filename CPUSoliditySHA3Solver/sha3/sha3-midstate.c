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

#include "sha3-midstate.h"

#if defined(__clang__)
#	define ROTL_64(x, n)		__builtin_rotateleft64(x, n)
#	define ROTR_64(x, n)		__builtin_rotateright64(x, n)
#else
#	define ROTL_64(x, n)		((x << n) | (x >> (64 - n)))
#	define ROTR_64(x, n)		((x >> n) | (x << (64 - n)))
#endif

#define CHI(a, b, c)			(a ^ ((~b) & c))
#define XOR5(a, b, c, d, e)		(a ^ b ^ c ^ d ^ e)
#define XOR3(a, b, c)			(a ^ b ^ c)

static inline uint64_t bswap64(uint64_t const input)
{
	return
		((input << 56) & 0xff00000000000000ull) |
		((input << 40) & 0x00ff000000000000ull) |
		((input << 24) & 0x0000ff0000000000ull) |
		((input <<  8) & 0x000000ff00000000ull) |
		((input >>  8) & 0x00000000ff000000ull) |
		((input >> 24) & 0x0000000000ff0000ull) |
		((input >> 40) & 0x000000000000ff00ull) |
		((input >> 56) & 0x00000000000000ffull);
}

void sha3_midstate(uint64_t const *midState, uint64_t const target, uint64_t const workPosition,
					uint32_t const maxSolutionCount, uint32_t *solutionCount, uint64_t *solutions)
{
	uint64_t state[25], C[5], D[5], n[11];
	uint64_t const rc[24] =
	{
		0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
		0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
		0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
		0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
		0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
		0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
		0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
		0x8000000000008080, 0x0000000080000001, 0x8000000080008008
	};

	n[0] = ROTL_64(workPosition, 7);
	n[1] = ROTL_64(n[0], 1);
	n[2] = ROTL_64(n[1], 6);
	n[3] = ROTL_64(n[2], 2);
	n[4] = ROTL_64(n[3], 4);
	n[5] = ROTL_64(n[4], 7);
	n[6] = ROTL_64(n[5], 12);
	n[7] = ROTL_64(n[6], 5);
	n[8] = ROTL_64(n[7], 11);
	n[9] = ROTL_64(n[8], 7);
	n[10] = ROTL_64(n[9], 1);

	C[0] = midState[0];
	C[1] = midState[1];
	C[2] = midState[2] ^ n[7];
	C[3] = midState[3];
	C[4] = midState[4] ^ n[2];
	state[0] = CHI(C[0], C[1], C[2]) ^ rc[0];
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = midState[5];
	C[1] = midState[6] ^ n[4];
	C[2] = midState[7];
	C[3] = midState[8];
	C[4] = midState[9] ^ n[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = midState[10];
	C[1] = midState[11] ^ n[0];
	C[2] = midState[12];
	C[3] = midState[13] ^ n[1];
	C[4] = midState[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = midState[15] ^ n[5];
	C[1] = midState[16];
	C[2] = midState[17];
	C[3] = midState[18] ^ n[3];
	C[4] = midState[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = midState[20] ^ n[10];
	C[1] = midState[21] ^ n[8];
	C[2] = midState[22] ^ n[6];
	C[3] = midState[23];
	C[4] = midState[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[1];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[2];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[3];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[4];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[5];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[6];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[7];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[8];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[9];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[10];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[11];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[12];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[13];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[14];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[15];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[16];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[17];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[18];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[19];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[20];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[21];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	state[0] = XOR3(state[0], D[0], C[0]);
	state[5] = XOR3(state[5], D[0], C[0]);
	state[10] = XOR3(state[10], D[0], C[0]);
	state[15] = XOR3(state[15], D[0], C[0]);
	state[20] = XOR3(state[20], D[0], C[0]);

	D[1] = ROTL_64(C[3], 1);
	state[1] = XOR3(state[1], D[1], C[1]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[11] = XOR3(state[11], D[1], C[1]);
	state[16] = XOR3(state[16], D[1], C[1]);
	state[21] = XOR3(state[21], D[1], C[1]);

	D[2] = ROTL_64(C[4], 1);
	state[2] = XOR3(state[2], D[2], C[2]);
	state[7] = XOR3(state[7], D[2], C[2]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[17] = XOR3(state[17], D[2], C[2]);
	state[22] = XOR3(state[22], D[2], C[2]);

	D[3] = ROTL_64(C[0], 1);
	state[3] = XOR3(state[3], D[3], C[3]);
	state[8] = XOR3(state[8], D[3], C[3]);
	state[13] = XOR3(state[13], D[3], C[3]);
	state[18] = XOR3(state[18], D[3], C[3]);
	state[23] = XOR3(state[23], D[3], C[3]);

	D[4] = ROTL_64(C[1], 1);
	state[4] = XOR3(state[4], D[4], C[4]);
	state[9] = XOR3(state[9], D[4], C[4]);
	state[14] = XOR3(state[14], D[4], C[4]);
	state[19] = XOR3(state[19], D[4], C[4]);
	state[24] = XOR3(state[24], D[4], C[4]);

	C[0] = state[1];
	state[1] = ROTR_64(state[6], 20);
	state[6] = ROTL_64(state[9], 20);
	state[9] = ROTR_64(state[22], 3);
	state[22] = ROTR_64(state[14], 25);
	state[14] = ROTL_64(state[20], 18);
	state[20] = ROTR_64(state[2], 2);
	state[2] = ROTR_64(state[12], 21);
	state[12] = ROTL_64(state[13], 25);
	state[13] = ROTL_64(state[19], 8);
	state[19] = ROTR_64(state[23], 8);
	state[23] = ROTR_64(state[15], 23);
	state[15] = ROTL_64(state[4], 27);
	state[4] = ROTL_64(state[24], 14);
	state[24] = ROTL_64(state[21], 2);
	state[21] = ROTR_64(state[8], 9);
	state[8] = ROTR_64(state[16], 19);
	state[16] = ROTR_64(state[5], 28);
	state[5] = ROTL_64(state[3], 28);
	state[3] = ROTL_64(state[18], 21);
	state[18] = ROTL_64(state[17], 15);
	state[17] = ROTL_64(state[11], 10);
	state[11] = ROTL_64(state[7], 6);
	state[7] = ROTL_64(state[10], 3);
	state[10] = ROTL_64(C[0], 1);

	C[0] = state[0];
	C[1] = state[1];
	C[2] = state[2];
	C[3] = state[3];
	C[4] = state[4];
	state[0] = CHI(C[0], C[1], C[2]);
	state[1] = CHI(C[1], C[2], C[3]);
	state[2] = CHI(C[2], C[3], C[4]);
	state[3] = CHI(C[3], C[4], C[0]);
	state[4] = CHI(C[4], C[0], C[1]);

	C[0] = state[5];
	C[1] = state[6];
	C[2] = state[7];
	C[3] = state[8];
	C[4] = state[9];
	state[5] = CHI(C[0], C[1], C[2]);
	state[6] = CHI(C[1], C[2], C[3]);
	state[7] = CHI(C[2], C[3], C[4]);
	state[8] = CHI(C[3], C[4], C[0]);
	state[9] = CHI(C[4], C[0], C[1]);

	C[0] = state[10];
	C[1] = state[11];
	C[2] = state[12];
	C[3] = state[13];
	C[4] = state[14];
	state[10] = CHI(C[0], C[1], C[2]);
	state[11] = CHI(C[1], C[2], C[3]);
	state[12] = CHI(C[2], C[3], C[4]);
	state[13] = CHI(C[3], C[4], C[0]);
	state[14] = CHI(C[4], C[0], C[1]);

	C[0] = state[15];
	C[1] = state[16];
	C[2] = state[17];
	C[3] = state[18];
	C[4] = state[19];
	state[15] = CHI(C[0], C[1], C[2]);
	state[16] = CHI(C[1], C[2], C[3]);
	state[17] = CHI(C[2], C[3], C[4]);
	state[18] = CHI(C[3], C[4], C[0]);
	state[19] = CHI(C[4], C[0], C[1]);

	C[0] = state[20];
	C[1] = state[21];
	C[2] = state[22];
	C[3] = state[23];
	C[4] = state[24];
	state[20] = CHI(C[0], C[1], C[2]);
	state[21] = CHI(C[1], C[2], C[3]);
	state[22] = CHI(C[2], C[3], C[4]);
	state[23] = CHI(C[3], C[4], C[0]);
	state[24] = CHI(C[4], C[0], C[1]);

	state[0] = state[0] ^ rc[22];

	C[1] = XOR5(state[0], state[5], state[10], state[15], state[20]);
	C[2] = XOR5(state[1], state[6], state[11], state[16], state[21]);
	C[3] = XOR5(state[2], state[7], state[12], state[17], state[22]);
	C[4] = XOR5(state[3], state[8], state[13], state[18], state[23]);
	C[0] = XOR5(state[4], state[9], state[14], state[19], state[24]);

	D[0] = ROTL_64(C[2], 1);
	D[1] = ROTL_64(C[3], 1);
	D[2] = ROTL_64(C[4], 1);

	state[0] = XOR3(state[0], D[0], C[0]);
	state[6] = XOR3(state[6], D[1], C[1]);
	state[12] = XOR3(state[12], D[2], C[2]);
	state[6] = ROTR_64(state[6], 20);
	state[12] = ROTR_64(state[12], 21);

	state[0] = CHI(state[0], state[6], state[12]) ^ rc[23];

	if (bswap64(state[0]) <= target) // LTE is allowed because target is high 64 bits of uint256
	{
		if (*solutionCount < maxSolutionCount)
		{
			solutions[*solutionCount] = workPosition;
			(*solutionCount)++;
		}
	}
}