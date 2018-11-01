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
