using System;

namespace SoliditySHA3Miner.NetworkInterface
{
    public delegate void GetMiningParameterStatusEvent(INetworkInterface sender, bool success, MiningParameters miningParameters);
    public delegate void NewMessagePrefixEvent(INetworkInterface sender, string messagePrefix);
    public delegate void NewTargetEvent(INetworkInterface sender, string target);

    public interface INetworkInterface : IDisposable
    {
        event GetMiningParameterStatusEvent OnGetMiningParameterStatusEvent;
        event NewMessagePrefixEvent OnNewMessagePrefixEvent;
        event NewTargetEvent OnNewTargetEvent;

        bool IsPool { get; }
        ulong SubmittedShares { get; }
        ulong RejectedShares { get; }
        ulong Difficulty { get; }
        string DifficultyHex { get; }
        
        void UpdateMiningParameters();

        void SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, Miner.IMiner sender);
    }
}
