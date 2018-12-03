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
using Nethereum.Util;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class SlaveInterface : NetworkInterfaceBase
    {
        public bool IsPause { get; private set; }

        private readonly bool m_isPool;
        private bool m_isGetMiningParameters;

        #region NetworkInterfaceBase

        public override bool IsPool => m_isPool;

        public override event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public override event NewChallengeEvent OnNewChallenge;
        public override event NewTargetEvent OnNewTarget;
        public override event NewDifficultyEvent OnNewDifficulty;
        public override event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;
        public override event GetTotalHashrateEvent OnGetTotalHashrate;

        public override bool SubmitSolution(string address, byte[] digest, byte[] challenge, HexBigInteger difficulty, byte[] nonce, object sender)
        {
            m_challengeReceiveDateTime = DateTime.Now;
            var startSubmitDateTime = DateTime.Now;

            bool success = false, submitted = false;
            int retryCount = 0, maxRetries = 10;
            do
            {
                try
                {
                    lock (this)
                    {
                        if (IsPause) return false;

                        if (SubmittedShares == ulong.MaxValue)
                        {
                            SubmittedShares = 0;
                            RejectedShares = 0;
                        }

                        Program.Print(string.Format("[INFO] Submitting solution to master URL({0})...", SubmitURL));

                        JObject submitSolution = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.SubmitSolution,
                                                                                          Utils.Numerics.Byte32ArrayToHexString(digest),
                                                                                          Utils.Numerics.Byte32ArrayToHexString(challenge),
                                                                                          Utils.Numerics.Byte32ArrayToHexString(difficulty.Value.ToByteArray(isUnsigned: true, isBigEndian: true)),
                                                                                          Utils.Numerics.Byte32ArrayToHexString(nonce));

                        var response = Utils.Json.InvokeJObjectRPC(SubmitURL, submitSolution, customTimeout: 10 * 60);

                        LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        if (!IsChallengedSubmitted(challenge))
                        {
                            m_submittedChallengeList.Insert(0, challenge.ToArray());
                            if (m_submittedChallengeList.Count > 100) m_submittedChallengeList.Remove(m_submittedChallengeList.Last());
                        }

                        var result = response.SelectToken("$.result")?.Value<string>();

                        success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        SubmittedShares++;
                        submitted = true;

                        if (success)
                        {
                            if (m_submitDateTimeList.Count > MAX_SUBMIT_DTM_COUNT)
                                m_submitDateTimeList.RemoveAt(0);

                            m_submitDateTimeList.Add(DateTime.Now);
                        }
                        else
                        {
                            RejectedShares++;
                        }

                        Program.Print(string.Format("[INFO] Nonce [{0}] submitted to master URL({1}): {2} ({3}ms)",
                                                    SubmittedShares,
                                                    SubmitURL,
                                                    (success ? "success" : "failed"),
                                                    LastSubmitLatency));
#if DEBUG
                        Program.Print(submitSolution.ToString());
                        Program.Print(response.ToString());
#endif
                        UpdateMiningParameters();
                    }
                }
                catch (Exception ex)
                {
                    retryCount += 1;

                    if (retryCount >= Math.Min(maxRetries, 3))
                        HandleException(ex, "Master not receiving nonce:");

                    if (retryCount < maxRetries)
                        Task.Delay(500);
                }
            } while (!submitted && retryCount < maxRetries);

            return success;
        }

        protected override void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                var totalHashRate = 0ul;
                OnGetTotalHashrate(this, ref totalHashRate);
                Program.Print(string.Format("[INFO] Total Hashrate: {0} MH/s (Effective) / {1} MH/s (Local)",
                                            GetEffectiveHashrate() / 1000000.0f, totalHashRate / 1000000.0f));

                var timeLeftToSolveBlock = GetTimeLeftToSolveBlock(totalHashRate);

                if (timeLeftToSolveBlock.TotalSeconds < 0)
                {
                    Program.Print(string.Format("[INFO] Estimated time left to solution: -({0}d {1}h {2}m {3}s)",
                                                Math.Abs(timeLeftToSolveBlock.Days),
                                                Math.Abs(timeLeftToSolveBlock.Hours),
                                                Math.Abs(timeLeftToSolveBlock.Minutes),
                                                Math.Abs(timeLeftToSolveBlock.Seconds)));
                }
                else
                {
                    Program.Print(string.Format("[INFO] Estimated time left to solution: {0}d {1}h {2}m {3}s",
                                                Math.Abs(timeLeftToSolveBlock.Days),
                                                Math.Abs(timeLeftToSolveBlock.Hours),
                                                Math.Abs(timeLeftToSolveBlock.Minutes),
                                                Math.Abs(timeLeftToSolveBlock.Seconds)));
                }
            }
            catch { }
        }

        protected override void UpdateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            if (m_isGetMiningParameters) return;
            try
            {
                m_isGetMiningParameters = true;
                var miningParameters = GetMiningParameters();
                if (miningParameters == null)
                {
                    OnGetMiningParameterStatus(this, false);
                    return;
                }

                CurrentChallenge = miningParameters.ChallengeByte32;

                if (m_lastParameters == null || miningParameters.Challenge.Value != m_lastParameters.Challenge.Value)
                {
                    Program.Print(string.Format("[INFO] New challenge detected {0}...", miningParameters.ChallengeByte32String));
                    OnNewChallenge(this, miningParameters.ChallengeByte32, MinerAddress);

                    if (m_challengeReceiveDateTime == DateTime.MinValue)
                        m_challengeReceiveDateTime = DateTime.Now;
                }

                if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                {
                    Program.Print(string.Format("[INFO] New target detected {0}...", miningParameters.MiningTargetByte32String));
                    OnNewTarget(this, miningParameters.MiningTarget);
                }

                if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                {
                    Program.Print(string.Format("[INFO] New difficulity detected ({0})...", miningParameters.MiningDifficulty.Value));
                    OnNewDifficulty?.Invoke(this, miningParameters.MiningDifficulty);
                    Difficulty = miningParameters.MiningDifficulty;

                    // Actual difficulty should have decimals
                    var calculatedDifficulty = BigDecimal.Exp(BigInteger.Log(MaxTarget.Value) - BigInteger.Log(miningParameters.MiningTarget.Value));
                    var calculatedDifficultyBigInteger = BigInteger.Parse(calculatedDifficulty.ToString().Split(",.".ToCharArray())[0]);

                    try // Perform rounding
                    {
                        if (uint.Parse(calculatedDifficulty.ToString().Split(",.".ToCharArray())[1].First().ToString()) >= 5)
                            calculatedDifficultyBigInteger++;
                    }
                    catch { }

                    if (Difficulty.Value != calculatedDifficultyBigInteger)
                    {
                        Difficulty = new HexBigInteger(calculatedDifficultyBigInteger);
                        var expValue = BigInteger.Log10(calculatedDifficultyBigInteger);
                        var calculatedTarget = BigInteger.Parse(
                            (BigDecimal.Parse(MaxTarget.Value.ToString()) * BigDecimal.Pow(10, expValue) / (calculatedDifficulty * BigDecimal.Pow(10, expValue))).
                            ToString().Split(",.".ToCharArray())[0]);
                        var calculatedTargetHex = new HexBigInteger(calculatedTarget);

                        Program.Print(string.Format("[INFO] Update target 0x{0}...", calculatedTarget.ToString("x64")));
                        OnNewTarget(this, calculatedTargetHex);
                        CurrentTarget = calculatedTargetHex;
                    }
                }

                IsPause = miningParameters.IsPause;
                m_lastParameters = miningParameters;

                OnStopSolvingCurrentChallenge(this, stopSolving: miningParameters.IsPause);
                OnGetMiningParameterStatus(this, true);
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }
            finally { m_isGetMiningParameters = false; }
        }

        #endregion

        public SlaveInterface(string masterURL, int updateInterval, int hashratePrintInterval)
            : base(updateInterval, hashratePrintInterval)
        {
            m_isGetMiningParameters = false;
            SubmitURL = masterURL;

            Program.Print(string.Format("[INFO] Waiting for master instance ({0}) to start...", SubmitURL));

            var getMasterAddress = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetMinerAddress);
            var getMaximumTarget = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetMaximumTarget);
            var getKingAddress = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetKingAddress);
            var getPoolMining = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetPoolMining);

            var retryCount = 0;
            while (true)
                try
                {
                    var parameters = MiningParameters.GetMiningParameters(SubmitURL,
                                                                          getEthAddress: getMasterAddress,
                                                                          getMaximumTarget: getMaximumTarget,
                                                                          getKingAddress: getKingAddress,
                                                                          getPoolMining: getPoolMining);
                    m_isPool = parameters.IsPoolMining;
                    MinerAddress = parameters.EthAddress;
                    MaxTarget = parameters.MaximumTarget;
                    Miner.Work.KingAddress = parameters.KingAddressByte20;
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount > 3)
                        HandleException(ex);

                    Task.Delay(1000).Wait();
                }

            if (hashratePrintInterval > 0)
            {
                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += HashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();
            }
        }

        private MiningParameters GetMiningParameters()
        {
            Program.Print(string.Format("[INFO] Checking latest parameters from master URL: {0}", SubmitURL));

            var getChallenge = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetChallenge);
            var getDifficulty = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetDifficulty);
            var getTarget = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetTarget);
            var getPause = Miner.MasterInterface.GetMasterParameter(Miner.MasterInterface.RequestMethods.GetPause);

            var success = true;
            var startTime = DateTime.Now;
            try
            {
                return MiningParameters.GetMiningParameters(SubmitURL,
                                                            getChallenge: getChallenge,
                                                            getDifficulty: getDifficulty,
                                                            getTarget: getTarget,
                                                            getPause: getPause);
            }
            catch (Exception ex)
            {
                HandleException(ex);
                success = false;
            }
            finally
            {
                if (success)
                    Latency = (int)(DateTime.Now - startTime).TotalMilliseconds;
            }
            return null;
        }

        private void HandleException(Exception ex, string errorPrefix = null)
        {
            var errorMessage = new StringBuilder("[ERROR] Occured at Slave instance => ");

            if (!string.IsNullOrWhiteSpace(errorPrefix))
                errorMessage.AppendFormat("{0}: ", errorPrefix);

            errorMessage.Append(ex.Message);

            var innerEx = ex.InnerException;
            while (innerEx != null)
            {
                errorMessage.AppendFormat("\n {0}", innerEx.Message);
                innerEx = innerEx.InnerException;
            }
            Program.Print(errorMessage.ToString());
        }
    }
}