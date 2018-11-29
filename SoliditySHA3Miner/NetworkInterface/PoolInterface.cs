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
using System.Net.NetworkInformation;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class PoolInterface : INetworkInterface
    {
        private readonly BigInteger uint256_MaxValue = BigInteger.Pow(2, 256);
        private DateTime m_challengeReceiveDateTime;

        private const int MAX_SUBMIT_DTM_COUNT = 50;
        private readonly List<DateTime> m_submitDateTimeList;

        private readonly string s_PoolURL;
        private readonly HexBigInteger m_customDifficulity;
        private readonly int m_maxScanRetry;
        private readonly int m_updateInterval;
        private bool m_isGetMiningParameters;
        private Timer m_updateMinerTimer;
        private Timer m_hashPrintTimer;
        private bool m_runFailover;
        private int m_retryCount;
        private MiningParameters m_lastParameters;

        public event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public event NewChallengeEvent OnNewChallenge;
        public event NewTargetEvent OnNewTarget;
        public event NewDifficultyEvent OnNewDifficulty;

#   pragma warning disable 67 // Unused event
        public event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;
#   pragma warning restore 67

        public event GetTotalHashrateEvent OnGetTotalHashrate;

        public bool IsPool => true;
        public bool IsSecondaryPool { get; }
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public PoolInterface SecondaryPool { get; }
        public HexBigInteger Difficulty { get; private set; }
        public HexBigInteger MaxTarget { get; private set; }
        public int LastSubmitLatency { get; private set; }
        public int Latency { get; private set; }
        public string MinerAddress { get; }
        public HexBigInteger CurrentTarget { get; private set; }

        public string SubmitURL
        {
            get
            {
                return m_runFailover
                    ? SecondaryPool.SubmitURL
                    : s_PoolURL;
            }
        }

        public byte[] CurrentChallenge { get; private set; }

        public PoolInterface(string minerAddress, string poolURL, int maxScanRetry, int updateInterval, int hashratePrintInterval,
                             BigInteger customDifficulity, bool isSecondary, HexBigInteger maxTarget, PoolInterface secondaryPool = null)
        {
            m_retryCount = 0;
            m_maxScanRetry = maxScanRetry;
            m_updateInterval = updateInterval;
            m_isGetMiningParameters = false;
            m_customDifficulity = new HexBigInteger(customDifficulity);
            Difficulty = new HexBigInteger((customDifficulity > 0) ? customDifficulity : 0);
            MaxTarget = maxTarget;
            LastSubmitLatency = -1;
            Latency = -1;
            SecondaryPool = secondaryPool;
            IsSecondaryPool = isSecondary;

            MinerAddress = minerAddress;
            s_PoolURL = poolURL;
            SubmittedShares = 0ul;
            RejectedShares = 0ul;

            m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT + 1);

            if (hashratePrintInterval > 0)
            {
                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += HashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();
            }
        }

        public void Dispose()
        {
            if (SecondaryPool != null) SecondaryPool.Dispose();

            if (m_submitDateTimeList != null)
                m_submitDateTimeList.Clear();

            if (m_updateMinerTimer != null)
            {
                try
                {
                    m_updateMinerTimer.Elapsed -= UpdateMinerTimer_Elapsed;
                    m_updateMinerTimer.Stop();
                    m_updateMinerTimer.Dispose();
                }
                catch { }
                m_updateMinerTimer = null;
            }

            if (m_hashPrintTimer != null)
            {
                try
                {
                    m_hashPrintTimer.Elapsed -= HashPrintTimer_Elapsed;
                    m_hashPrintTimer.Stop();
                    m_hashPrintTimer.Dispose();
                }
                catch { }
                m_hashPrintTimer = null;
            }
        }

        private JObject GetPoolParameter(string method, params string[] parameters)
        {
            var paramObject = new JObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = "1",
                ["method"] = method
            };
            if (parameters != null)
            {
                if (parameters.Length > 0)
                {
                    JArray props = new JArray();
                    foreach (var p in parameters) { props.Add(p); }
                    paramObject.Add(new JProperty("params", props));
                }
            }
            return paramObject;
        }

        public MiningParameters GetMiningParameters()
        {
            Program.Print(string.Format("[INFO] Checking latest parameters from {0} pool...", IsSecondaryPool ? "secondary" : "primary"));

            var getPoolEthAddress = GetPoolParameter("getPoolEthAddress");
            var getPoolChallengeNumber = GetPoolParameter("getChallengeNumber");
            var getPoolMinimumShareDifficulty = GetPoolParameter("getMinimumShareDifficulty", MinerAddress);
            var getPoolMinimumShareTarget = GetPoolParameter("getMinimumShareTarget", MinerAddress);

            var success = true;
            var startTime = DateTime.Now;
            try
            {
                return MiningParameters.GetMiningParameters(s_PoolURL,
                                                            getEthAddress: getPoolEthAddress,
                                                            getChallenge: getPoolChallengeNumber,
                                                            getDifficulty: getPoolMinimumShareDifficulty,
                                                            getTarget: getPoolMinimumShareTarget);
            }
            catch (Exception ex)
            {
                m_retryCount++;
                success = false;
                var errorMessage = new StringBuilder("[ERROR] " + ex.Message);

                var innerEx = ex.InnerException;
                while (innerEx != null)
                {
                    errorMessage.AppendFormat("\n {0}", innerEx.Message);
                    innerEx = innerEx.InnerException;
                }

                Program.Print(errorMessage.ToString());
            }
            finally
            {
                if (success)
                {
                    m_runFailover = false;
                    var tempLatency = (int)(DateTime.Now - startTime).TotalMilliseconds;
                    try
                    {
                        using (var ping = new Ping())
                        {
                            var poolURL = s_PoolURL.Contains("://") ? s_PoolURL.Split(new string[] { "://" }, StringSplitOptions.None)[1] : s_PoolURL;
                            try
                            {
                                var response = ping.Send(poolURL);
                                if (response.RoundtripTime > 0)
                                    tempLatency = (int)response.RoundtripTime;
                            }
                            catch
                            {
                                try
                                {
                                    poolURL = poolURL.Split('/').First();
                                    var response = ping.Send(poolURL);
                                    if (response.RoundtripTime > 0)
                                        tempLatency = (int)response.RoundtripTime;
                                }
                                catch
                                {
                                    try
                                    {
                                        poolURL = poolURL.Split(':').First();
                                        var response = ping.Send(poolURL);
                                        if (response.RoundtripTime > 0)
                                            tempLatency = (int)response.RoundtripTime;
                                    }
                                    catch { }
                                }
                            }
                        }
                    }
                    catch { }
                    Latency = tempLatency;
                }
            }

            var runFailover = (!success && SecondaryPool != null && m_maxScanRetry > -1 && m_retryCount >= m_maxScanRetry);
            try
            {
                if (runFailover) return SecondaryPool.GetMiningParameters(); 
            }
            finally { m_runFailover = runFailover; }

            return null;
        }

        private void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var totalHashRate = 0ul;
            try
            {
                OnGetTotalHashrate(this, ref totalHashRate);
                Program.Print(string.Format("[INFO] Total Hashrate: {0} MH/s (Effective) / {1} MH/s (Local)",
                                            GetEffectiveHashrate() / 1000000.0f, totalHashRate / 1000000.0f));
            }
            catch (Exception)
            {
                try
                {
                    totalHashRate = GetEffectiveHashrate();
                    Program.Print(string.Format("[INFO] Effective Hashrate: {0} MH/s", totalHashRate / 1000000.0f));
                }
                catch { }
            }
            try
            {
                if (totalHashRate > 0)
                {
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
            }
            catch { }
        }

        private void UpdateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
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
                    OnNewChallenge(this, miningParameters.ChallengeByte32, miningParameters.EthAddress);

                    if (m_challengeReceiveDateTime == DateTime.MinValue)
                        m_challengeReceiveDateTime = DateTime.Now;
                }

                if (m_customDifficulity.Value == 0)
                {
                    if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                    {
                        Program.Print(string.Format("[INFO] New target detected {0}...", miningParameters.MiningTargetByte32String));
                        OnNewTarget(this, miningParameters.MiningTarget);
                        CurrentTarget = miningParameters.MiningTarget;
                    }

                    if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                    {
                        Program.Print(string.Format("[INFO] New difficulity detected ({0})...", miningParameters.MiningDifficulty.Value));
                        OnNewDifficulty?.Invoke(this, miningParameters.MiningDifficulty);
                        Difficulty = miningParameters.MiningDifficulty;

                        // Actual difficulty should have decimals
                        var calculatedDifficulty = BigDecimal.Exp(BigInteger.Log(MaxTarget.Value) - BigInteger.Log(miningParameters.MiningTarget.Value));
                        var calculatedDifficultyBigInteger = BigInteger.Parse(calculatedDifficulty.ToString().Split(",.".ToCharArray())[0]);

                        // Only replace if the integer portion is different
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
                }
                else
                {
                    Difficulty = new HexBigInteger(m_customDifficulity);
                    var calculatedTarget = MaxTarget.Value / m_customDifficulity;
                    var newTarget = new HexBigInteger(calculatedTarget);

                    OnNewTarget(this, newTarget);
                }

                m_lastParameters = miningParameters;
                OnGetMiningParameterStatus(this, true);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
            finally { m_isGetMiningParameters = false; }
        }

        /// <summary>
        /// <para>Since a single hash is a random number between 1 and 2^256, and difficulty [1] target = 2^234</para>
        /// <para>Then we can find difficulty [N] target = 2^234 / N</para>
        /// <para>Hence, # of hashes to find block with difficulty [N] = N * 2^256 / 2^234</para>
        /// <para>Which simplifies to # of hashes to find block difficulty [N] = N * 2^22</para>
        /// <para>Time to find block in seconds with difficulty [N] = N * 2^22 / hashes per second</para>
        /// </summary>
        public TimeSpan GetTimeLeftToSolveBlock(ulong hashrate)
        {
            if (MaxTarget == null || MaxTarget.Value == 0 || Difficulty == null || Difficulty.Value == 0 || hashrate == 0 || m_challengeReceiveDateTime == DateTime.MinValue)
                return TimeSpan.Zero;

            var timeToSolveBlock = new BigInteger(Difficulty) * uint256_MaxValue / MaxTarget.Value / new BigInteger(hashrate);

            var secondsLeftToSolveBlock = timeToSolveBlock - (long)(DateTime.Now - m_challengeReceiveDateTime).TotalSeconds;

            return (secondsLeftToSolveBlock > (long)TimeSpan.MaxValue.TotalSeconds)
                ? TimeSpan.MaxValue
                : TimeSpan.FromSeconds((long)secondsLeftToSolveBlock);
        }

        /// <summary>
        /// <para>Since a single hash is a random number between 1 and 2^256, and difficulty [1] target = 2^234</para>
        /// <para>Then we can find difficulty [N] target = 2^234 / N</para>
        /// <para>Hence, # of hashes to find block with difficulty [N] = N * 2^256 / 2^234</para>
        /// <para>Which simplifies to # of hashes to find block difficulty [N] = N * 2^22</para>
        /// <para>Time to find block in seconds with difficulty [N] = N * 2^22 / hashes per second</para>
        /// <para>Hashes per second with difficulty [N] and time to find block [T] = N * 2^22 / T</para>
        /// </summary>
        public ulong GetEffectiveHashrate()
        {
            var hashrate = 0ul;

            if (m_submitDateTimeList.Count > 1)
            {
                var avgSolveTime = (ulong)((DateTime.Now - m_submitDateTimeList.First()).TotalSeconds / m_submitDateTimeList.Count - 1);
                hashrate = (ulong)(Difficulty.Value * uint256_MaxValue / MaxTarget.Value / new BigInteger(avgSolveTime));
            }

            return hashrate;
        }

        public void ResetEffectiveHashrate()
        {
            m_submitDateTimeList.Clear();
            m_submitDateTimeList.Add(DateTime.Now);
        }

        public void UpdateMiningParameters()
        {
            UpdateMinerTimer_Elapsed(this, null);

            if (m_updateMinerTimer == null && m_updateInterval > 0)
            {
                m_updateMinerTimer = new Timer(m_updateInterval);
                m_updateMinerTimer.Elapsed += UpdateMinerTimer_Elapsed;
                m_updateMinerTimer.Start();
            }
        }

        public bool SubmitSolution(string address, byte[] digest, byte[] challenge, HexBigInteger difficulty, byte[] nonce, object sender)
        {
            m_challengeReceiveDateTime = DateTime.Now;
            var startSubmitDateTime = DateTime.Now;

            if (m_runFailover)
            {
                if (SecondaryPool.SubmitSolution(address, digest, challenge, difficulty, nonce, sender))
                {
                    LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);
                    Program.Print(string.Format("[INFO] Submission roundtrip latency: {0}ms", LastSubmitLatency));

                    if (m_submitDateTimeList.Count >= MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                    m_submitDateTimeList.Add(DateTime.Now);
                }
                return false;
            }

            bool success = false, submitted = false;
            int retryCount = 0, maxRetries = 10;
            var devFee = (long)Math.Round(100 / Math.Abs(DevFee.UserPercent));
            const long devShareOffset = 10;
            do
            {
                try
                {
                    lock (this)
                    {
                        if (SubmittedShares == ulong.MaxValue)
                        {
                            SubmittedShares = 0;
                            RejectedShares = 0;
                        }
                        var minerAddress = (((long)(SubmittedShares - RejectedShares) - devShareOffset) % devFee) == 0
                                         ? DevFee.Address
                                         : MinerAddress;

                        JObject submitShare = GetPoolParameter("submitShare",
                                                               Utils.Numerics.Byte32ArrayToHexString(nonce),
                                                               minerAddress,
                                                               Utils.Numerics.Byte32ArrayToHexString(digest),
                                                               difficulty.Value.ToString(),
                                                               Utils.Numerics.Byte32ArrayToHexString(challenge),
                                                               m_customDifficulity.Value > 0 ? "true" : "false",
                                                               Miner.Work.GetKingAddressString());

                        var response = Utils.Json.InvokeJObjectRPC(s_PoolURL, submitShare);

                        LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        var result = response.SelectToken("$.result")?.Value<string>();

                        success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        SubmittedShares++;
                        submitted = true;

                        Program.Print(string.Format("[INFO] {0} [{1}] submitted to {2} pool: {3} ({4}ms)",
                                                    (minerAddress == DevFee.Address ? "Dev. fee share" : "Miner share"),
                                                    SubmittedShares,
                                                    IsSecondaryPool ? "secondary" : "primary",
                                                    (success ? "success" : "failed"),
                                                    LastSubmitLatency));
                        if (success)
                        {
                            if (m_submitDateTimeList.Count > MAX_SUBMIT_DTM_COUNT)
                                m_submitDateTimeList.RemoveAt(0);

                            m_submitDateTimeList.Add(DateTime.Now);
                        }
                        else
                        {
                            RejectedShares++;
                            UpdateMiningParameters();
                        }
#if DEBUG
                        Program.Print(submitShare.ToString());
                        Program.Print(response.ToString());
#endif
                    }
                }
                catch (Exception ex)
                {
                    retryCount += 1;

                    if (retryCount >= Math.Min(maxRetries, 3))
                        Program.Print(string.Format("[ERROR] Pool not receiving nonce: {0}", ex.Message));

                    if (retryCount < maxRetries)
                        Task.Delay(500);
                }
            } while (!submitted && retryCount < maxRetries);

            return success;
        }
    }
}