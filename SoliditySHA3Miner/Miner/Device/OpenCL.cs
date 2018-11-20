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

using Newtonsoft.Json;

namespace SoliditySHA3Miner.Miner.Device
{
    public class OpenCL : DeviceBase
    {
        public const float DEFAULT_INTENSITY = 24.056f;
        public const float DEFAULT_INTENSITY_KING = 24.12f;
        public const float DEFAULT_INTENSITY_INTEL = 17.0f;
        public const ulong DEFAULT_LOCAL_WORK_SIZE = 128;
        public const ulong DEFAULT_LOCAL_WORK_SIZE_INTEL = 64;

        [JsonIgnore]
        public Structs.DeviceCL DeviceCL_Struct;

        [JsonIgnore]
        public ulong[] Solutions;
    }
}
