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
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace SoliditySHA3Miner.Structs
{
    [StructLayout(LayoutKind.Sequential)]
    public struct DeviceCUDA
    {
        public int DeviceID;
        public int PciBusID;
        public IntPtr Name;
        public int ComputeMajor;
        public int ComputeMinor;
        public float Intensity;
        public ulong Threads;
        public Dim3 Grid;
        public Dim3 Block;
        public ulong WorkPosition;
        public uint MaxSolutionCount;
        public IntPtr SolutionCount;
        public IntPtr SolutionCountDevice;
        public IntPtr Solutions;
        public IntPtr SolutionsDevice;
        public Instance InstanceCUDA;

        public unsafe string NameToString()
        {
            byte* namePtr = (byte*)Name.ToPointer();
            if (namePtr == null) return null;
            return string.Concat(Encoding.ASCII.GetString(namePtr, 256).TakeWhile(c => c != '\0').ToArray());
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Instance
        {
            public IntPtr API;
        }
    }
}