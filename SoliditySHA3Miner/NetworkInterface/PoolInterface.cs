using Nethereum.Hex.HexTypes;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.NetworkInformation;
using System.Numerics;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class PoolInterface : INetworkInterface
    {
        /// <summary>
        /// <para>Since a single hash is a random number between 1 and 2^256, and difficulty [1] target = 2^234</para>
        /// <para>Then we can find difficulty [N] target = 2^234 / N</para>
        /// <para>Hence, # of hashes to find block with difficulty [N] = N * 2^256 / 2^234</para>
        /// <para>Which simplifies to # of hashes to find block difficulty [N] = N * 2^22</para>
        /// <para>Time to find block in seconds with difficulty [N] = N * 2^22 / hashes per second</para>
        /// <para>Hashes per second with difficulty [N] and time to find block [T] = N * 2^22 / T</para>
        /// </summary>
        private readonly ulong EFFECTIVE_HASHRATE_CONST = (ulong)Math.Pow(2, 22);

        private const int MAX_SUBMIT_DTM_COUNT = 50;
        private readonly List<DateTime> m_submitDateTimeList;

        private readonly string s_PoolURL;
        private readonly ulong m_customDifficulity;
        private readonly HexBigInteger m_maxTarget;
        private readonly int m_maxScanRetry;
        private readonly int m_updateInterval;
        private bool m_isGetMiningParameters;
        private Timer m_updateMinerTimer;
        private Timer m_hashPrintTimer;
        private bool m_runFailover;
        private int m_retryCount;
        private MiningParameters m_lastParameters;
        
        public event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public event NewMessagePrefixEvent OnNewMessagePrefix;
        public event NewTargetEvent OnNewTarget;
        public event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;

        public event GetTotalHashrateEvent OnGetTotalHashrate;

        public bool IsPool => true;
        public bool IsSecondaryPool { get; }
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public PoolInterface SecondaryPool { get; }
        public ulong Difficulty { get; private set; }
        public string DifficultyHex { get; private set; }
        public int LastSubmitLatency { get; private set; }
        public int Latency { get; private set; }
        public string MinerAddress { get; }

        public string SubmitURL
        {
            get
            {
                if (m_runFailover)
                    return SecondaryPool.SubmitURL;
                else
                    return s_PoolURL;
            }
        }

        public string CurrentChallenge { get; private set; }

        public PoolInterface(string minerAddress, string poolURL, int maxScanRetry, int updateInterval, int hashratePrintInterval,
                             ulong customDifficulity, bool isSecondary, HexBigInteger maxTarget, PoolInterface secondaryPool = null)
        {
            m_retryCount = 0;
            m_maxScanRetry = maxScanRetry;
            m_customDifficulity = customDifficulity;
            m_maxTarget = maxTarget;
            m_updateInterval = updateInterval;
            m_isGetMiningParameters = false;
            LastSubmitLatency = -1;
            Latency = -1;
            SecondaryPool = secondaryPool;
            IsSecondaryPool = isSecondary;

            if (m_customDifficulity > 0)
                Difficulty = m_customDifficulity;

            MinerAddress = minerAddress;
            s_PoolURL = poolURL;
            SubmittedShares = 0ul;
            RejectedShares = 0ul;

            m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT + 1);

            if (hashratePrintInterval > 0)
            {
                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += m_hashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();
            }
        }

        public void Dispose()
        {
            if (SecondaryPool != null) SecondaryPool.Dispose();
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

            bool success = true;
            var startTime = DateTime.Now;
            try
            {
                return MiningParameters.GetPoolMiningParameters(s_PoolURL, getPoolEthAddress, getPoolChallengeNumber,
                                                                getPoolMinimumShareDifficulty, getPoolMinimumShareTarget);
            }
            catch (AggregateException ex)
            {
                success = false;
                m_retryCount++;

                Program.Print("[ERROR] " + ex.Message);
            }
            catch (Exception ex)
            {
                success = false;
                m_retryCount++;

                string errorMsg = ex.Message;
                if (ex.InnerException != null) errorMsg += ("\n " + ex.InnerException.Message);
                Program.Print("[ERROR] " + errorMsg);
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
                            var poolURL = s_PoolURL.Contains("://") ? s_PoolURL.Split("://")[1] : s_PoolURL;
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

        private void m_hashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var totalHashRate = 0ul;
            OnGetTotalHashrate(this, ref totalHashRate);
            Program.Print(string.Format("[INFO] Total Hashrate: {0} MH/s (Effective) / {1} MH/s (Local)",
                                        GetEffectiveHashrate() / 1000000.0f, totalHashRate / 1000000.0f));
        }

        private void m_updateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            if (m_isGetMiningParameters) return;
            try
            {
                m_isGetMiningParameters = true;
                var miningParameters = GetMiningParameters();
                if (miningParameters == null)
                {
                    OnGetMiningParameterStatus(this, false, null);
                    return;
                }

                var address = miningParameters.EthAddress;
                var target = miningParameters.MiningTargetByte32String;
                CurrentChallenge = miningParameters.ChallengeNumberByte32String;

                if (m_lastParameters == null || miningParameters.ChallengeNumber.Value != m_lastParameters.ChallengeNumber.Value)
                {
                    Program.Print(string.Format("[INFO] New challenge detected {0}...", CurrentChallenge));
                    OnNewMessagePrefix(this, CurrentChallenge + address.Replace("0x", string.Empty));
                }

                if (m_customDifficulity == 0)
                {
                    DifficultyHex = miningParameters.MiningDifficulty.HexValue;

                    if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                    {
                        Program.Print(string.Format("[INFO] New target detected {0}...", target));
                        OnNewTarget(this, target);
                    }

                    if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                    {
                        Program.Print(string.Format("[INFO] New difficulity detected ({0})...", miningParameters.MiningDifficulty.Value));
                        Difficulty = Convert.ToUInt64(miningParameters.MiningDifficulty.Value.ToString());

                        var calculatedTarget = m_maxTarget.Value / Difficulty;
                        if (calculatedTarget != miningParameters.MiningTarget.Value)
                        {
                            var newTarget = calculatedTarget.ToString();
                            Program.Print(string.Format("[INFO] Update target {0}...", newTarget));
                            OnNewTarget(this, newTarget);
                        }
                    }
                }
                else
                {
                    Difficulty = m_customDifficulity;
                    var calculatedTarget = m_maxTarget.Value / m_customDifficulity;
                    var newTarget = new HexBigInteger(new BigInteger(m_customDifficulity)).HexValue;

                    OnNewTarget(this, newTarget);
                }

                m_lastParameters = miningParameters;
                OnGetMiningParameterStatus(this, true, miningParameters);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
            finally { m_isGetMiningParameters = false; }
        }

        public ulong GetEffectiveHashrate()
        {
            var hashrate = 0ul;

            if (m_submitDateTimeList.Count > 1)
            {
                var avgSolveTime = (ulong)((DateTime.Now - m_submitDateTimeList.First()).TotalSeconds / m_submitDateTimeList.Count - 1);
                hashrate = Difficulty * EFFECTIVE_HASHRATE_CONST / avgSolveTime;
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
            m_updateMinerTimer_Elapsed(this, null);

            if (m_updateMinerTimer == null && m_updateInterval > 0)
            {
                m_updateMinerTimer = new Timer(m_updateInterval);
                m_updateMinerTimer.Elapsed += m_updateMinerTimer_Elapsed;
                m_updateMinerTimer.Start();
            }
        }

        public bool SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, Miner.IMiner sender)
        {
            if (string.IsNullOrWhiteSpace(solution) || solution == "0x") return false;

            var startSubmitDateTime = DateTime.Now;

            if (m_runFailover)
            {
                if (SecondaryPool.SubmitSolution(digest, fromAddress, challenge, difficulty, target, solution, sender))
                {
                    LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);
                    Program.Print(string.Format("[INFO] Submission roundtrip latency: {0}ms", LastSubmitLatency));

                    if (m_submitDateTimeList.Count >= MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                    m_submitDateTimeList.Add(DateTime.Now);
                }
                return false;
            }

            difficulty = new HexBigInteger(difficulty).Value.ToString(); // change from hex to base 10 numerics

            var success = false;
            var submitted = false;
            int retryCount = 0, maxRetries = 10;
            var devFee = (ulong)Math.Round(100 / Math.Abs(DevFee.UserPercent));
            do
            {
                try
                {
                    var poolAddress = fromAddress;
                    lock (this)
                    {
                        if (SubmittedShares == ulong.MaxValue) SubmittedShares = 0u;
                        var minerAddress = ((SubmittedShares) % devFee) == 0 ? DevFee.Address : MinerAddress;

                        JObject submitShare;
                        submitShare = GetPoolParameter("submitShare", solution, minerAddress, digest, difficulty, challenge,
                                                       m_customDifficulity > 0 ? "true" : "false", Miner.Work.GetKingAddressString());

                        var response = Utils.Json.InvokeJObjectRPC(s_PoolURL, submitShare);

                        LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        var result = response.SelectToken("$.result")?.Value<string>();

                        success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        if (!success) RejectedShares++;
                        SubmittedShares++;

                        Program.Print(string.Format("[INFO] {0} [{1}] submitted to {2} pool: {3} ({4}ms)",
                                                    (minerAddress == DevFee.Address ? "Dev. fee share" : "Miner share"),
                                                    SubmittedShares,
                                                    IsSecondaryPool ? "secondary" : "primary",
                                                    (success ? "success" : "failed"),
                                                    LastSubmitLatency));
#if DEBUG
                        Program.Print(submitShare.ToString());
                        Program.Print(response.ToString());
#endif
                    }
                    submitted = true;
                }
                catch (Exception ex)
                {
                    Program.Print(string.Format("[ERROR] {0}", ex.Message));

                    retryCount += 1;
                    if (retryCount < maxRetries) Task.Delay(500);
                }
            } while (!submitted && retryCount < maxRetries);

            if (success)
            {
                if (m_submitDateTimeList.Count > MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                m_submitDateTimeList.Add(DateTime.Now);
            }
            else UpdateMiningParameters();

            return success;
        }
    }
}