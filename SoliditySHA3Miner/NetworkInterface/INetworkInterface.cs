using System;

namespace SoliditySHA3Miner.NetworkInterface
{
    public interface INetworkInterface : IDisposable
    {
        bool IsPool { get; }
        ulong SubmittedShares { get; }
        ulong RejectedShares { get; }

        MiningParameters GetMiningParameters();
        void SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, bool isCustomDifficulty);
    }
}
