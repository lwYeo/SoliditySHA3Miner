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
using SoliditySHA3Miner.Structs;
using System;

namespace SoliditySHA3Miner.Miner.Device
{
    public class CUDA : DeviceBase
    {
        public const float DEFAULT_INTENSITY = 24.0f;

        private float m_lastIntensity;
        private uint m_lastThreads;
        private uint m_lastCompute;
        private Dim3 m_lastBlock;
        private Dim3 m_newBlock;
        private Dim3 m_gridSize;

        [JsonIgnore]
        public Structs.DeviceCUDA DeviceCUDA_Struct;

        [JsonIgnore]
        public uint ConputeVersion;

        [JsonIgnore]
        public uint Threads
        {
            get
            {
                if (ConputeVersion <= 500)
                    Intensity = Intensity <= 40.55f ? Intensity : 40.55f;

                if (Intensity != m_lastIntensity)
                {
                    m_lastThreads = (uint)Math.Pow(2, Intensity);
                    m_lastIntensity = Intensity;
                    m_lastBlock.X = 0;
                }
                return m_lastThreads;
            }
        }

        [JsonIgnore]
        public Dim3 Block
        {
            get
            {
                if (m_lastCompute != ConputeVersion)
                {
                    m_lastCompute = ConputeVersion;

                    switch (ConputeVersion)
                    {
                        case 520:
                        case 610:
                        case 700:
                        case 720:
                        case 750:
                            m_newBlock.X = 1024u;
                            break;
                        case 300:
                        case 320:
                        case 350:
                        case 370:
                        case 500:
                        case 530:
                        case 600:
                        case 620:
                        default:
                            m_newBlock.X = (ConputeVersion >= 800) ? 1024u : 384u;
                            break;
                    }
                }
                return m_newBlock;
            }
        }

        [JsonIgnore]
        public Dim3 Grid
        {
            get
            {
                if (m_lastBlock.X != Block.X)
                {
                    m_gridSize.X = (Threads + Block.X - 1) / Block.X;
                    m_lastBlock.X = Block.X;
                }
                return m_gridSize;
            }
        }

        public CUDA()
        {
            m_lastBlock.X = 1;
            m_lastBlock.Y = 1;
            m_lastBlock.Z = 1;

            m_newBlock.X = 1;
            m_newBlock.Y = 1;
            m_newBlock.Z = 1;

            m_gridSize.X = 1;
            m_gridSize.Y = 1;
            m_gridSize.Z = 1;
        }
    }
}