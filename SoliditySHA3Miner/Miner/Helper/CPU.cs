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

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace SoliditySHA3Miner.Miner.Helper
{
    public static class CPU
    {
        #region P/Invoke interface

        public static class Solver
        {
            public const string SOLVER_NAME = "CPUSoliditySHA3Solver";

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void SHA3(IntPtr message, IntPtr digest);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetCpuName(StringBuilder cpuName);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern IntPtr GetInstance();

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void DisposeInstance(IntPtr instance);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void SetThreadAffinity(IntPtr instance, int affinityMask, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void HashMessage(IntPtr instance, ref Structs.DeviceCPU device, ref Structs.Processor processor);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void HashMidState(IntPtr instance, ref Structs.DeviceCPU device, ref Structs.Processor processor);
        }

        #endregion

        private static readonly Random RandomGenerator =
            new Random(
                BitConverter.ToInt32(
                    BitConverter.GetBytes(
                        BitConverter.ToUInt32(BitConverter.GetBytes(Guid.NewGuid().GetHashCode()))
                        + BitConverter.ToUInt32(BitConverter.GetBytes(DateTime.Now.GetHashCode())))));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong ROTL64(ulong input, int offset)
        {
            return ((input << offset) ^ (input >> (64 - offset)));
        }

        public static byte[] GetMidState(byte[] challenge, byte[] address, byte[] solutionTemplate)
        {
            var C = (ulong[])Array.CreateInstance(typeof(ulong), 5);
            var D = (ulong[])Array.CreateInstance(typeof(ulong), 5);
            var midStateBytes = (byte[])Array.CreateInstance(typeof(byte), MinerBase.SPONGE_LENGTH);
            var midStateU64 = (ulong[])Array.CreateInstance(typeof(ulong), MinerBase.SPONGE_LENGTH / MinerBase.UINT64_LENGTH);
            var messageBytes = (byte[])Array.CreateInstance(typeof(byte), MinerBase.MESSAGE_LENGTH);

            Array.ConstrainedCopy(challenge, 0, messageBytes, 0, MinerBase.UINT256_LENGTH);
            Array.ConstrainedCopy(address, 0, messageBytes, MinerBase.UINT256_LENGTH, MinerBase.ADDRESS_LENGTH);
            Array.ConstrainedCopy(solutionTemplate, 0, messageBytes, MinerBase.UINT256_LENGTH + MinerBase.ADDRESS_LENGTH, MinerBase.UINT256_LENGTH);

            var messageU64 = (ulong[])Array.CreateInstance(typeof(ulong), (int)Math.Round(((double)MinerBase.MESSAGE_LENGTH / MinerBase.UINT64_LENGTH), MidpointRounding.AwayFromZero));
            Buffer.BlockCopy(messageBytes, 0, messageU64, 0, MinerBase.MESSAGE_LENGTH);

            C[0] = messageU64[0] ^ messageU64[5] ^ messageU64[10] ^ 0x100000000ul;
            C[1] = messageU64[1] ^ messageU64[6] ^ 0x8000000000000000ul;
            C[2] = messageU64[2] ^ messageU64[7];
            C[3] = messageU64[3] ^ messageU64[8];
            C[4] = messageU64[4] ^ messageU64[9];

            D[0] = ROTL64(C[1], 1) ^ C[4];
            D[1] = ROTL64(C[2], 1) ^ C[0];
            D[2] = ROTL64(C[3], 1) ^ C[1];
            D[3] = ROTL64(C[4], 1) ^ C[2];
            D[4] = ROTL64(C[0], 1) ^ C[3];

            midStateU64[0] = messageU64[0] ^ D[0];
            midStateU64[1] = ROTL64(messageU64[6] ^ D[1], 44);
            midStateU64[2] = ROTL64(D[2], 43);
            midStateU64[3] = ROTL64(D[3], 21);
            midStateU64[4] = ROTL64(D[4], 14);
            midStateU64[5] = ROTL64(messageU64[3] ^ D[3], 28);
            midStateU64[6] = ROTL64(messageU64[9] ^ D[4], 20);
            midStateU64[7] = ROTL64(messageU64[10] ^ D[0] ^ 0x100000000ul, 3);
            midStateU64[8] = ROTL64(0x8000000000000000ul ^ D[1], 45);
            midStateU64[9] = ROTL64(D[2], 61);
            midStateU64[10] = ROTL64(messageU64[1] ^ D[1], 1);
            midStateU64[11] = ROTL64(messageU64[7] ^ D[2], 6);
            midStateU64[12] = ROTL64(D[3], 25);
            midStateU64[13] = ROTL64(D[4], 8);
            midStateU64[14] = ROTL64(D[0], 18);
            midStateU64[15] = ROTL64(messageU64[4] ^ D[4], 27);
            midStateU64[16] = ROTL64(messageU64[5] ^ D[0], 36);
            midStateU64[17] = ROTL64(D[1], 10);
            midStateU64[18] = ROTL64(D[2], 15);
            midStateU64[19] = ROTL64(D[3], 56);
            midStateU64[20] = ROTL64(messageU64[2] ^ D[2], 62);
            midStateU64[21] = ROTL64(messageU64[8] ^ D[3], 55);
            midStateU64[22] = ROTL64(D[4], 39);
            midStateU64[23] = ROTL64(D[0], 41);
            midStateU64[24] = ROTL64(D[1], 2);

            Buffer.BlockCopy(midStateU64, 0, midStateBytes, 0, MinerBase.SPONGE_LENGTH);
            return midStateBytes;
        }

        public static byte[] GetSolutionTemplate(string kingAddress = null)
        {
            var template = (byte[])Array.CreateInstance(typeof(byte), MinerBase.UINT256_LENGTH);
            RandomGenerator.NextBytes(template);

            if (string.IsNullOrWhiteSpace(kingAddress))
                for (var i = (MinerBase.UINT256_LENGTH / 2 - MinerBase.UINT64_LENGTH / 2); i < (MinerBase.UINT256_LENGTH / 2 + MinerBase.UINT64_LENGTH / 2); i++)
                    template[i] = 0;
            else
            {
                Work.KingAddress = (byte[])Array.CreateInstance(typeof(byte), MinerBase.ADDRESS_LENGTH);
                Utils.Numerics.AddressStringToByte20Array(kingAddress, ref Work.KingAddress, type: "king address");

                for (var i = 0; i < MinerBase.ADDRESS_LENGTH; i++)
                    template[i] = Work.KingAddress[i];

                for (var i = MinerBase.ADDRESS_LENGTH; i < (MinerBase.ADDRESS_LENGTH + MinerBase.UINT64_LENGTH); i++)
                    template[i] = 0;
            }
            return template;
        }
    }
}