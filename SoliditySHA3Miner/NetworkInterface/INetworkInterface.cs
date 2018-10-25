using System;

namespace SoliditySHA3Miner.NetworkInterface
{
    public delegate void GetMiningParameterStatusEvent(INetworkInterface sender, bool success, MiningParameters miningParameters);
    public delegate void NewMessagePrefixEvent(INetworkInterface sender, string messagePrefix);
    public delegate void NewTargetEvent(INetworkInterface sender, string target);
    public delegate void StopSolvingCurrentChallengeEvent(INetworkInterface sender, string currentTarget);

    public delegate void GetTotalHashrateEvent(INetworkInterface sender, ref ulong totalHashrate);

    public interface INetworkInterface : IDisposable
    {
        event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        event NewMessagePrefixEvent OnNewMessagePrefix;
        event NewTargetEvent OnNewTarget;
        event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;

        event GetTotalHashrateEvent OnGetTotalHashrate;

        bool IsPool { get; }
        ulong SubmittedShares { get; }
        ulong RejectedShares { get; }
        ulong Difficulty { get; }
        string DifficultyHex { get; }
        int LastSubmitLatency { get; }
        int Latency { get; }
        string MinerAddress { get; }
        string SubmitURL { get; }
        string CurrentChallenge { get; }

        TimeSpan GetTimeLeftToSolveBlock(ulong hashrate);
        ulong GetEffectiveHashrate();
        void ResetEffectiveHashrate();
        void UpdateMiningParameters();

        bool SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, Miner.IMiner sender);
    }
}