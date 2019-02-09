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
    public struct DeviceCL
    {
        public int Enum;
        public int PciBusID;
        public IntPtr Name;
        public IntPtr CL_ID;
        public ulong Type;
        public Platform PlatformCL;
        public bool IsAMD;
        public ulong MaxWorkGroupSize;
        public ulong GlobalWorkSize;
        public ulong LocalWorkSize;
        public float Intensity;
        public ulong WorkPosition;
        public uint MaxSolutionCount;
        public uint SolutionCount;
        public IntPtr Solutions;
        public IntPtr InstanceCL;

        public unsafe string NameToString()
        {
            byte* namePtr = (byte*)Name.ToPointer();
            if (namePtr == null) return null;
            return string.Concat(Encoding.ASCII.GetString(namePtr, 256).TakeWhile(c => c != '\0').ToArray());
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Platform
        {
            public IntPtr ID;
            public IntPtr Name;

            public unsafe string NameToString()
            {
                byte* namePtr = (byte*)Name.ToPointer();
                return string.Concat(Encoding.ASCII.GetString(namePtr, 256).TakeWhile(c => c != '\0').ToArray());
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Instance
        {
            IntPtr API;

            IntPtr MessageBuffer;
            IntPtr SolutionCountBuffer;
            IntPtr SolutionsBuffer;
            IntPtr MidStateBuffer;
            IntPtr TargetBuffer;

            IntPtr Queue;
            IntPtr Context;
            IntPtr Program;
            IntPtr Kernel;

            IntPtr KernelWaitEvent;
            IntPtr KernelWaitSleepDuration;

            IntPtr Solutions;
            IntPtr SolutionCount;
        }
    }
}