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

#include <array>

static const unsigned short UINT32_LENGTH{ 4u };
static const unsigned short UINT64_LENGTH{ 8u };
static const unsigned short MIDSTATE_LENGTH{ 25u };
static const unsigned short SPONGE_LENGTH{ 200u };
static const unsigned short ADDRESS_LENGTH{ 20u };
static const unsigned short UINT256_LENGTH{ 32u };
static const unsigned short PREFIX_LENGTH{ ADDRESS_LENGTH + UINT256_LENGTH }; // challenge32 + address20
static const unsigned short MESSAGE_LENGTH{ PREFIX_LENGTH + UINT256_LENGTH }; // challenge32 + address20 + solution32

typedef std::array<uint8_t, ADDRESS_LENGTH> address_t;
typedef std::array<uint8_t, UINT256_LENGTH> byte32_t;
typedef std::array<uint8_t, PREFIX_LENGTH> prefix_t; // challenge32 + address20
typedef std::array<uint8_t, MESSAGE_LENGTH> message_t; // challenge32 + address20 + solution32
typedef std::array<uint8_t, SPONGE_LENGTH> sponge_t;

typedef struct _message_s
{
	byte32_t				challenge;
	address_t				address;
	byte32_t				solution;
} message_s;

typedef union _message_ut
{
	message_t				byteArray;
	message_s				structure;
} message_ut;

typedef union _sponge_ut
{
	sponge_t				byteArray;
	uint64_t				uint64Array[25];
} sponge_ut;