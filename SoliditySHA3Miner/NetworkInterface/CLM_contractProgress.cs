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

using Nethereum.ABI.FunctionEncoding.Attributes;
using System.Numerics;

namespace SoliditySHA3Miner.NetworkInterface
{
    [FunctionOutput]
    public class CLM_ContractProgress
    {
        [Parameter("uint256", "epoch", 1)]
        public BigInteger Epoch { get; set; }

        [Parameter("uint256", "candidate", 2)]
        public BigInteger Candidate { get; set; }

        [Parameter("uint256", "round", 3)]
        public BigInteger Round { get; set; }

        [Parameter("uint256", "miningepoch", 4)]
        public BigInteger MiningEpoch { get; set; }

        [Parameter("uint256", "globalreward", 5)]
        public BigInteger GlobalReward { get; set; }

        [Parameter("uint256", "powreward", 6)]
        public BigInteger PowReward { get; set; }

        [Parameter("uint256", "masternodereward", 7)]
        public BigInteger MasternodeReward { get; set; }

        [Parameter("uint256", "usercounter", 8)]
        public BigInteger UserCounter { get; set; }
    }
}
