using System;
using System.Numerics;
using System.Threading.Tasks;
using System.Timers;
using Nethereum.Hex.HexTypes;
using Newtonsoft.Json.Linq;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class PoolInterface : INetworkInterface
    {
        private readonly string s_MinerAddress;
        private readonly string s_PoolURL;
        private readonly uint m_customDifficulity;
        private readonly HexBigInteger m_maxTarget;
        private readonly int m_maxScanRetry;
        private readonly int m_updateInterval;
        private Timer m_updateMinerTimer;
        private bool m_runFailover;
        private int m_retryCount;
        private MiningParameters m_lastParameters;
        private MiningParameters m_cacheParameters;

        public event GetMiningParameterStatusEvent OnGetMiningParameterStatusEvent;
        public event NewMessagePrefixEvent OnNewMessagePrefixEvent;
        public event NewTargetEvent OnNewTargetEvent;

        public bool IsPool => true;
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public PoolInterface SecondaryPool { get; }
        public ulong Difficulty { get; private set; }
        public string DifficultyHex { get; private set; }

        public PoolInterface(string minerAddress, string poolURL, int maxScanRetry, int updateInterval, uint customDifficulity,
                             HexBigInteger maxTarget, PoolInterface secondaryPool = null)
        {
            m_retryCount = 0;
            m_maxScanRetry = maxScanRetry;
            m_customDifficulity = customDifficulity;
            m_maxTarget = maxTarget;
            m_updateInterval = updateInterval;
            SecondaryPool = secondaryPool;

            if (m_customDifficulity > 0)
                Difficulty = m_customDifficulity;

            s_MinerAddress = minerAddress;
            s_PoolURL = poolURL;
            SubmittedShares = 0ul;
            RejectedShares = 0ul;
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
            Program.Print("[INFO] Checking latest parameters from pool...");

            var getPoolEthAddress = GetPoolParameter("getPoolEthAddress");
            var getPoolChallengeNumber = GetPoolParameter("getChallengeNumber");
            var getPoolMinimumShareDifficulty = GetPoolParameter("getMinimumShareDifficulty", s_MinerAddress);
            var getPoolMinimumShareTarget = GetPoolParameter("getMinimumShareTarget", s_MinerAddress);

            bool success = true;
            try
            {
                m_cacheParameters = MiningParameters.GetPoolMiningParameters(s_PoolURL, getPoolEthAddress, getPoolChallengeNumber,
                                                                             getPoolMinimumShareDifficulty, getPoolMinimumShareTarget);

                return m_cacheParameters;
            }
            catch (AggregateException ex)
            {
                success = false;
                m_retryCount++;

                string errorMsg = ex.Message;
                foreach (var iEx in ex.InnerExceptions) errorMsg += ("\n " + iEx.Message);

                Program.Print("[ERROR] " + errorMsg);
            }
            catch (Exception ex)
            {
                success = false;
                m_retryCount++;

                string errorMsg = ex.Message;
                if (ex.InnerException != null) errorMsg += ("\n " + ex.InnerException.Message);
                Program.Print("[ERROR] " + errorMsg);
            }

            if (!success && SecondaryPool != null && m_retryCount >= m_maxScanRetry)
            {
                m_runFailover = true;
                Program.Print("[INFO] Checking mining parameters from secondary pool...");
                return SecondaryPool.GetMiningParameters();
            }

            return null;
        }

        private void m_updateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                var miningParameters = GetMiningParameters();
                if (miningParameters == null)
                {
                    OnGetMiningParameterStatusEvent(this, false, null);
                    return;
                }

                var address = miningParameters.EthAddress;
                var challenge = miningParameters.ChallengeNumberByte32String;
                var target = miningParameters.MiningTargetByte32String;

                if (m_lastParameters == null || miningParameters.ChallengeNumber.Value != m_lastParameters.ChallengeNumber.Value)
                {
                    Program.Print(string.Format("[INFO] New challenge detected {0}...", challenge));
                    OnNewMessagePrefixEvent(this, challenge + address.Replace("0x", string.Empty));
                }

                if (m_customDifficulity == 0)
                {
                    DifficultyHex = miningParameters.MiningDifficulty.HexValue;

                    if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                    {
                        Program.Print(string.Format("[INFO] New target detected {0}...", target));
                        OnNewTargetEvent(this, target);
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
                            OnNewTargetEvent(this, newTarget);
                        }
                    }
                }
                else
                {
                    Difficulty = m_customDifficulity;
                    var calculatedTarget = m_maxTarget.Value / m_customDifficulity;
                    var newTarget = new HexBigInteger(new BigInteger(m_customDifficulity)).HexValue;

                    OnNewTargetEvent(this, newTarget);
                }

                m_lastParameters = miningParameters;
                OnGetMiningParameterStatusEvent(this, true, miningParameters);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
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

        public void SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, Miner.IMiner sender)
        {
            if (m_runFailover)
            {
                ((INetworkInterface)SecondaryPool).SubmitSolution(digest, fromAddress, challenge, difficulty, target, solution, sender);
                return;
            }

            if (string.IsNullOrWhiteSpace(solution) || solution == "0x") return;

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
                        var minerAddress = ((SubmittedShares) % devFee) == 0 ? DevFee.Address : s_MinerAddress;

                        JObject submitShare;
                        submitShare = GetPoolParameter("submitShare", solution, minerAddress, digest, difficulty, challenge,
                                                       m_customDifficulity > 0 ? "true" : "false");

                        var response = Utils.Json.InvokeJObjectRPC(s_PoolURL, submitShare);
                        var result = response.SelectToken("$.result")?.Value<string>();

                        success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        if (!success) RejectedShares++;
                        SubmittedShares++;

                        Program.Print(string.Format("[INFO] {0} [{1}] submitted: {2}", 
                                                    (minerAddress == DevFee.Address ? "Dev. fee share" : "Miner share"),
                                                    SubmittedShares, 
                                                    (success ? "success" : "failed")));
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

            if (!success) UpdateMiningParameters();
        }
    }
}
