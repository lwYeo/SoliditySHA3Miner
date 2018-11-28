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

using System.Runtime.CompilerServices;

namespace SoliditySHA3Miner.Miner
{
    public static class Work
    {
        private static ulong m_Position = 0;

        private static readonly object m_positionLock = new object();

        public static byte[] KingAddress;

        public static byte[] SolutionTemplate;

        public static string GetKingAddressString()
        {
            if (KingAddress == null)
                return string.Empty;
            else
                return Utils.Numerics.Byte32ArrayToHexString(KingAddress, prefix: true);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void GetPosition(ref ulong workPosition)
        {
            lock (m_positionLock) { workPosition = m_Position; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ResetPosition(ref ulong lastPosition)
        {
            lock (m_positionLock)
            {
                lastPosition = m_Position;
                m_Position = 0;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void IncrementPosition(ref ulong lastPosition, ulong increment)
        {
            lock (m_positionLock)
            {
                lastPosition = m_Position;
                m_Position += increment;
            }
        }
    }
}
