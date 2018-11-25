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

using SoliditySHA3Miner.Structs;
using Newtonsoft.Json;
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace SoliditySHA3Miner.Miner.Device
{
    public abstract class DeviceBase : IDisposable
    {
        public const uint MAX_SOLUTION_COUNT = 4;

        public bool AllowDevice;
        public string Type;
        public string Platform;
        public int DeviceID;
        public uint PciBusID;
        public string Name;
        public float Intensity;

        [JsonIgnore]
        public bool IsAssigned;
        [JsonIgnore]
        public bool IsInitialized;
        [JsonIgnore]
        public bool IsMining;
        [JsonIgnore]
        public bool IsPause;
        [JsonIgnore]
        public bool IsStopped;
        [JsonIgnore]
        public bool HasNewTarget;
        [JsonIgnore]
        public bool HasNewChallenge;

        [JsonIgnore]
        public ulong[] High64Target;
        [JsonIgnore]
        public byte[] Target;
        [JsonIgnore]
        public byte[] Challenge;
        [JsonIgnore]
        public byte[] MidState;
        [JsonIgnore]
        public byte[] Message;

        [JsonIgnore]
        public ulong HashCount;
        [JsonIgnore]
        public DateTime HashStartTime;

        [JsonIgnore]
        public DeviceCommon CommonPointers;

        private readonly List<GCHandle> m_handles;

        public DeviceBase()
        {
            m_handles = new List<GCHandle>();

            High64Target = (ulong[])Array.CreateInstance(typeof(ulong), 1);
            Target = (byte[])Array.CreateInstance(typeof(byte), MinerBase.UINT256_LENGTH);
            Challenge = (byte[])Array.CreateInstance(typeof(byte), MinerBase.UINT256_LENGTH);
            MidState = (byte[])Array.CreateInstance(typeof(byte), MinerBase.SPONGE_LENGTH);
            Message = (byte[])Array.CreateInstance(typeof(byte), MinerBase.MESSAGE_LENGTH);

            CommonPointers.High64Target = AllocHandleAndGetPointer(High64Target);
            CommonPointers.Target = AllocHandleAndGetPointer(Target);
            CommonPointers.Challenge = AllocHandleAndGetPointer(Challenge);
            CommonPointers.MidState = AllocHandleAndGetPointer(MidState);
            CommonPointers.Message = AllocHandleAndGetPointer(Message);
        }

        public void AddHandle(GCHandle handle)
        {
            m_handles.Add(handle);
        }

        public void Dispose()
        {
            m_handles.Clear();
            m_handles.TrimExcess();
        }

        private IntPtr AllocHandleAndGetPointer(Array array)
        {
            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            m_handles.Add(handle);
            return handle.AddrOfPinnedObject();
        }
    }
}