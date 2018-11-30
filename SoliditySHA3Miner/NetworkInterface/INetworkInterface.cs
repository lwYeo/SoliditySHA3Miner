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

using Nethereum.Hex.HexTypes;
using System;

namespace SoliditySHA3Miner.NetworkInterface
{
    public delegate void GetMiningParameterStatusEvent(INetworkInterface sender, bool success);
    public delegate void NewChallengeEvent(INetworkInterface sender, byte[] challenge, string address);
    public delegate void NewTargetEvent(INetworkInterface sender, HexBigInteger target);
    public delegate void NewDifficultyEvent(INetworkInterface sender, HexBigInteger difficulty);
    public delegate void StopSolvingCurrentChallengeEvent(INetworkInterface sender, bool stopSolving = true);
    public delegate void GetTotalHashrateEvent(INetworkInterface sender, ref ulong totalHashrate);

    public interface INetworkInterface : IDisposable
    {
        bool IsPool { get; }

        ulong SubmittedShares { get; }

        ulong RejectedShares { get; }

        HexBigInteger Difficulty { get; }

        HexBigInteger MaxTarget { get; }

        int LastSubmitLatency { get; }

        int Latency { get; }

        string MinerAddress { get; }

        string SubmitURL { get; }

        byte[] CurrentChallenge { get; }

        HexBigInteger CurrentTarget { get; }

        event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        event NewChallengeEvent OnNewChallenge;
        event NewTargetEvent OnNewTarget;
        event NewDifficultyEvent OnNewDifficulty;
        event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;
        event GetTotalHashrateEvent OnGetTotalHashrate;

        bool SubmitSolution(string address, byte[] digest, byte[] challenge, HexBigInteger difficulty, byte[] nonce, object sender);

        ulong GetEffectiveHashrate();

        TimeSpan GetTimeLeftToSolveBlock(ulong hashrate);

        bool IsChallengedSubmitted(byte[] challenge);

        void ResetEffectiveHashrate();

        void UpdateMiningParameters();
    }
}