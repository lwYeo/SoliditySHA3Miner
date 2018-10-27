using Nethereum.Contracts;
using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using Nethereum.RPC.Eth.DTOs;
using Nethereum.Util;
using Nethereum.Web3;
using Nethereum.Web3.Accounts;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.NetworkInformation;
using System.Numerics;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class Web3Interface : INetworkInterface
    {
        private readonly BigInteger uint256_MaxValue = BigInteger.Pow(2, 256);
        private HexBigInteger m_maxTarget;
        private DateTime m_challengeReceiveDateTime;

        private const int MAX_TIMEOUT = 10;
        private const string DEFAULT_WEB3_API = Config.Defaults.InfuraAPI_mainnet;
        private const int MAX_SUBMIT_DTM_COUNT = 50;
        private readonly List<DateTime> m_submitDateTimeList;

        private readonly Web3 m_web3;
        private readonly Contract m_contract;
        private readonly Account m_account;
        private readonly Function m_mintMethod;
        private readonly Function m_transferMethod;
        private readonly Function m_getMiningDifficulty;
        private readonly Function m_getMiningTarget;
        private readonly Function m_getChallengeNumber;
        private readonly Function m_getMiningReward;
        private readonly Function m_MAXIMUM_TARGET;

        private readonly int m_mintMethodInputParamCount;

        private readonly float m_gasToMine;
        private readonly ulong m_gasLimit;
        private readonly List<string> m_submittedChallengeList;
        private readonly int m_updateInterval;
        private Timer m_updateMinerTimer;
        private Timer m_hashPrintTimer;
        private MiningParameters m_lastParameters;

        public event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public event NewMessagePrefixEvent OnNewMessagePrefix;
        public event NewTargetEvent OnNewTarget;
        public event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;

        public event GetTotalHashrateEvent OnGetTotalHashrate;

        public bool IsPool => false;
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public ulong Difficulty { get; private set; }
        public string DifficultyHex { get; private set; }
        public int LastSubmitLatency { get; private set; }
        public int Latency { get; private set; }
        public string MinerAddress { get; }
        public string SubmitURL { get; private set; }
        public string CurrentChallenge { get; private set; }

        public bool IsChallengedSubmitted(string challenge) => m_submittedChallengeList.Contains(challenge);

        public Web3Interface(string web3ApiPath, string contractAddress, string minerAddress, string privateKey,
                             float gasToMine, string abiFileName, int updateInterval, int hashratePrintInterval, ulong gasLimit)
        {
            m_updateInterval = updateInterval;
            m_submittedChallengeList = new List<string>();
            m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT + 1);
            Nethereum.JsonRpc.Client.ClientBase.ConnectionTimeout = MAX_TIMEOUT * 1000;
            LastSubmitLatency = -1;
            Latency = -1;

            if (string.IsNullOrWhiteSpace(contractAddress))
            {
                Program.Print("[INFO] Contract address not specified, default 0xBTC");
                contractAddress = Config.Defaults.Contract0xBTC_mainnet;
            }

            var addressUtil = new AddressUtil();
            if (!addressUtil.IsValidAddressLength(contractAddress))
            {
                throw new Exception("Invalid contract address provided, ensure address is 42 characters long (including '0x').");
            }
            else if (!addressUtil.IsChecksumAddress(contractAddress))
            {
                throw new Exception("Invalid contract address provided, ensure capitalization is correct.");
            }

            if (!string.IsNullOrWhiteSpace(privateKey))
            {
                m_account = new Account(privateKey);
                minerAddress = m_account.Address;
            }

            if (!addressUtil.IsValidAddressLength(minerAddress))
            {
                throw new Exception("Invalid miner address provided, ensure address is 42 characters long (including '0x').");
            }
            else if (!addressUtil.IsChecksumAddress(minerAddress))
            {
                throw new Exception("Invalid miner address provided, ensure capitalization is correct.");
            }

            Program.Print("[INFO] Miner's address : " + minerAddress);

            MinerAddress = minerAddress;
            SubmitURL = string.IsNullOrWhiteSpace(web3ApiPath) ? DEFAULT_WEB3_API : web3ApiPath;

            m_web3 = new Web3(SubmitURL);

            var erc20AbiPath = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), "ERC-20.abi");
            var tokenAbiPath = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), abiFileName);

            var erc20Abi = JArray.Parse(File.ReadAllText(erc20AbiPath));
            var tokenAbi = JArray.Parse(File.ReadAllText(tokenAbiPath));            
            tokenAbi.Merge(erc20Abi, new JsonMergeSettings { MergeArrayHandling = MergeArrayHandling.Union });

            m_contract = m_web3.Eth.GetContract(tokenAbi.ToString(), contractAddress);
            var contractABI = m_contract.ContractBuilder.ContractABI;

            if (!string.IsNullOrWhiteSpace(privateKey))
            {
                m_gasToMine = gasToMine;
                Program.Print(string.Format("[INFO] Gas to mine: {0}", m_gasToMine));

                m_gasLimit = gasLimit;
                Program.Print(string.Format("[INFO] Gas limit: {0}", m_gasLimit));

                m_mintMethod = m_contract.GetFunction("mint");
                m_transferMethod = m_contract.GetFunction("transfer");
                m_getMiningDifficulty = m_contract.GetFunction("getMiningDifficulty");
                m_getMiningTarget = m_contract.GetFunction("getMiningTarget");
                m_getChallengeNumber = m_contract.GetFunction("getChallengeNumber");
                m_getMiningReward = m_contract.GetFunction("getMiningReward");

                if (contractABI.Functions.Any(f => f.Name == "_MAXIMUM_TARGET"))
                    m_MAXIMUM_TARGET = m_contract.GetFunction("_MAXIMUM_TARGET");

                else if (contractABI.Functions.Any(f => f.Name == "MAX_TARGET"))
                    m_MAXIMUM_TARGET = m_contract.GetFunction("MAX_TARGET");

                m_mintMethodInputParamCount = m_contract.ContractBuilder.
                                                         ContractABI.Functions.
                                                         First(f => f.Name == "mint").
                                                         InputParameters.Count();

                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += m_hashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();
            }
        }

        public void Dispose()
        {
            m_submittedChallengeList.Clear();
            m_submittedChallengeList.TrimExcess();
        }

        public void OverrideMaxTarget(HexBigInteger maxTarget)
        {
            if (maxTarget.Value > 0u)
            {
                Program.Print("[INFO] Override maximum difficulty: " + maxTarget.HexValue);
                m_maxTarget = maxTarget;
            }
            else { m_maxTarget = GetMaxTarget(); }
        }

        public HexBigInteger GetMaxTarget()
        {
            if (m_maxTarget != null && m_maxTarget.Value > 0u)
                return m_maxTarget;

            Program.Print("[INFO] Checking maximum difficulity from network...");
            while (true)
            {
                try
                {
                    if (m_MAXIMUM_TARGET != null)
                        return new HexBigInteger(m_MAXIMUM_TARGET.CallAsync<BigInteger>().Result);
                    else
                        return new HexBigInteger("0x40000000000000000000000000000000000000000000000000000000000");
                }
                catch (AggregateException ex)
                {
                    var errorMessage = ex.Message;
                    var currentEx = ex.InnerExceptions[0] ?? ex.InnerException;

                    while (currentEx != null)
                    {
                        errorMessage += "\n " + currentEx.Message;
                        currentEx = currentEx.InnerException;
                    }
                    Program.Print("[ERROR] " + errorMessage);

                    System.Threading.Thread.Sleep(1000);
                }
            }
        }

        public MiningParameters GetMiningParameters()
        {
            Program.Print("[INFO] Checking latest parameters from network...");
            bool success = true;
            var startTime = DateTime.Now;
            try
            {
                return MiningParameters.GetSoloMiningParameters(MinerAddress, m_getMiningDifficulty, m_getMiningTarget, m_getChallengeNumber);
            }
            catch (Exception ex)
            {
                success = false;
                throw ex;
            }
            finally
            {
                if (success)
                {
                    var tempLatency = (int)(DateTime.Now - startTime).TotalMilliseconds;
                    try
                    {
                        using (var ping = new Ping())
                        {
                            var submitUrl = SubmitURL.Contains("://") ? SubmitURL.Split("://")[1] : SubmitURL;
                            try
                            {
                                var response = ping.Send(submitUrl);
                                if (response.RoundtripTime > 0)
                                    tempLatency = (int)response.RoundtripTime;
                            }
                            catch
                            {
                                try
                                {
                                    submitUrl = submitUrl.Split('/').First();
                                    var response = ping.Send(submitUrl);
                                    if (response.RoundtripTime > 0)
                                        tempLatency = (int)response.RoundtripTime;
                                }
                                catch
                                {
                                    try
                                    {
                                        submitUrl = submitUrl.Split(':').First();
                                        var response = ping.Send(submitUrl);
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
        }

        private void m_hashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var totalHashRate = 0ul;
            OnGetTotalHashrate(this, ref totalHashRate);
            Program.Print(string.Format("[INFO] Total Hashrate: {0} MH/s (Effective) / {1} MH/s (Local)",
                                        GetEffectiveHashrate() / 1000000.0f, totalHashRate / 1000000.0f));

            var timeLeftToSolveBlock = GetTimeLeftToSolveBlock(totalHashRate);
            var magnitude = (timeLeftToSolveBlock.TotalSeconds < 0) ? "- " : string.Empty;

            Program.Print(string.Format("[INFO] Estimated time left to solve block: {0}{1}d {2}h {3}m {4}s",
                                        magnitude,
                                        Math.Abs(timeLeftToSolveBlock.Days),
                                        Math.Abs(timeLeftToSolveBlock.Hours),
                                        Math.Abs(timeLeftToSolveBlock.Minutes),
                                        Math.Abs(timeLeftToSolveBlock.Seconds)));
        }

        private void m_updateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                var miningParameters = GetMiningParameters();
                if (miningParameters == null)
                {
                    OnGetMiningParameterStatus(this, false, null);
                    return;
                }

                var address = miningParameters.EthAddress;
                var target = miningParameters.MiningTargetByte32String;
                CurrentChallenge = miningParameters.ChallengeNumberByte32String;
                DifficultyHex = miningParameters.MiningDifficulty.HexValue;

                if (m_lastParameters == null || miningParameters.ChallengeNumber.Value != m_lastParameters.ChallengeNumber.Value)
                {
                    Program.Print(string.Format("[INFO] New challenge detected {0}...", CurrentChallenge));
                    OnNewMessagePrefix(this, CurrentChallenge + address.Replace("0x", string.Empty));
                    if (m_challengeReceiveDateTime == DateTime.MinValue) m_challengeReceiveDateTime = DateTime.Now;
                }

                if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                {
                    Program.Print(string.Format("[INFO] New target detected {0}...", target));
                    OnNewTarget(this, target);
                }

                if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                {
                    Program.Print(string.Format("[INFO] New difficulity detected ({0})...", miningParameters.MiningDifficulty.Value));
                    Difficulty = Convert.ToUInt64(miningParameters.MiningDifficulty.Value.ToString());

                    // Actual difficulty should have decimals
                    var calculatedDifficulty = Math.Exp(BigInteger.Log(m_maxTarget.Value) - BigInteger.Log(miningParameters.MiningTarget.Value));

                    if ((ulong)calculatedDifficulty != Difficulty) // Only replace if the integer portion is different
                    {
                        Difficulty = (ulong)calculatedDifficulty;

                        var expValue = BitConverter.GetBytes(decimal.GetBits((decimal)calculatedDifficulty)[3])[2];

                        var calculatedTarget = m_maxTarget.Value * (ulong)Math.Pow(10, expValue) / (ulong)(calculatedDifficulty * Math.Pow(10, expValue));
                        
                        Program.Print(string.Format("[INFO] Update target {0}...", Utils.Numerics.BigIntegerToByte32HexString(calculatedTarget)));
                        OnNewTarget(this, Utils.Numerics.BigIntegerToByte32HexString(calculatedTarget));
                    }
                }

                m_lastParameters = miningParameters;
                OnGetMiningParameterStatus(this, true, miningParameters);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
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
            if (m_maxTarget == null || m_maxTarget.Value == 0 || Difficulty == 0 || hashrate == 0 || m_challengeReceiveDateTime == DateTime.MinValue)
                return TimeSpan.Zero;

            var timeToSolveBlock = new BigInteger(Difficulty) * uint256_MaxValue / m_maxTarget.Value / new BigInteger(hashrate);

            var secondsLeftToSolveBlock = timeToSolveBlock -(long)(DateTime.Now - m_challengeReceiveDateTime).TotalSeconds;

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
                hashrate = (ulong)(new BigInteger(Difficulty) * uint256_MaxValue / m_maxTarget.Value / new BigInteger(avgSolveTime));
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
            lock (this)
            {
                if (IsChallengedSubmitted(challenge))
                {
                    OnStopSolvingCurrentChallenge(this, challenge);
                    Program.Print(string.Format("[INFO] Submission cancelled, nonce has been submitted for the current challenge."));
                    return false;
                }
                m_challengeReceiveDateTime = DateTime.MinValue;

                var transactionID = string.Empty;
                var gasLimit = new HexBigInteger(m_gasLimit);
                var userGas = new HexBigInteger(UnitConversion.Convert.ToWei(new BigDecimal(m_gasToMine), UnitConversion.EthUnit.Gwei));

                var oSolution = new BigInteger(Utils.Numerics.HexStringToByte32Array(solution).ToArray());
                // Note: do not directly use -> new HexBigInteger(solution).Value
                //Because two's complement representation always interprets the highest-order bit of the last byte in the array
                //(the byte at position Array.Length - 1) as the sign bit,
                //the method returns a byte array with an extra element whose value is zero
                //to disambiguate positive values that could otherwise be interpreted as having their sign bits set.

                object[] dataInput = null;

                if (m_mintMethodInputParamCount > 1) // 0xBitcoin compatibility
                    dataInput = new object[] { oSolution, HexByteConvertorExtensions.HexToByteArray(digest) };

                else // Draft EIP-918 compatibility [2018-03-07]
                    dataInput = new object[] { oSolution };

                var retryCount = 0u;
                var startSubmitDateTime = DateTime.Now;
                while (string.IsNullOrWhiteSpace(transactionID))
                {
                    try
                    {
                        var txCount = m_web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(fromAddress).Result;

                        var estimatedGasLimit = m_mintMethod.EstimateGasAsync(from: fromAddress,
                                                                              gas: gasLimit,
                                                                              value: new HexBigInteger(0),
                                                                              functionInput: dataInput).Result;

                        var transaction = m_mintMethod.CreateTransactionInput(from: fromAddress,
                                                                              gas: estimatedGasLimit,
                                                                              gasPrice: userGas,
                                                                              value: new HexBigInteger(0),
                                                                              functionInput: dataInput);

                        var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_account.PrivateKey,
                                                                                      to: m_contract.Address,
                                                                                      amount: 0,
                                                                                      nonce: txCount.Value,
                                                                                      gasPrice: userGas,
                                                                                      gasLimit: estimatedGasLimit,
                                                                                      data: transaction.Data);

                        if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                            throw new Exception("Failed to verify transaction.");

                        transactionID = m_web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;

                        LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        if (!string.IsNullOrWhiteSpace(transactionID))
                        {
                            if (!IsChallengedSubmitted(challenge))
                            {
                                m_submittedChallengeList.Insert(0, challenge);
                                if (m_submittedChallengeList.Count > 100) m_submittedChallengeList.Remove(m_submittedChallengeList.Last());
                            }

                            Task.Factory.StartNew(() => GetTransactionReciept(transactionID, fromAddress, gasLimit, userGas, LastSubmitLatency));
                        }
                    }
                    catch (AggregateException ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        foreach (var iEx in ex.InnerExceptions)
                            errorMessage += "\n " + iEx.Message;

                        Program.Print(errorMessage);
                        if (IsChallengedSubmitted(challenge)) return false;
                    }
                    catch (Exception ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        if (ex.InnerException != null)
                            errorMessage += "\n " + ex.InnerException.Message;

                        Program.Print(errorMessage);
                        if (IsChallengedSubmitted(challenge) || ex.Message == "Failed to verify transaction.") return false;
                    }

                    System.Threading.Thread.Sleep(1000);
                    if (string.IsNullOrWhiteSpace(transactionID)) retryCount++;

                    if (retryCount > 10)
                    {
                        Program.Print("[ERROR] Failed to submit solution for more than 10 times, please check settings.");
                        sender.StopMining();
                    }
                }
                if (m_submitDateTimeList.Count >= MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                m_submitDateTimeList.Add(DateTime.Now);

                return true;
            }
        }

        private void GetTransactionReciept(string transactionID, string fromAddress, HexBigInteger gasLimit, HexBigInteger userGas, int responseTime)
        {
            try
            {
                var success = false;
                TransactionReceipt reciept = null;
                while (reciept == null)
                {
                    try
                    {
                        System.Threading.Thread.Sleep(3000);
                        reciept = m_web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(transactionID).Result;
                    }
                    catch (AggregateException ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        foreach (var iEx in ex.InnerExceptions)
                            errorMessage += "\n " + iEx.Message;

                        Program.Print(errorMessage);
                    }
                    catch (Exception ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        if (ex.InnerException != null)
                            errorMessage += "\n " + ex.InnerException.Message;

                        Program.Print(errorMessage);
                    }
                }

                success = (reciept.Status.Value == 1);

                if (!success) RejectedShares++;
                if (SubmittedShares == ulong.MaxValue) SubmittedShares = 0ul;
                else SubmittedShares++;

                Program.Print(string.Format("[INFO] Miner share [{0}] submitted: {1} ({2}ms), block: {3}," +
                                            "\n transaction ID: {4}",
                                            SubmittedShares,
                                            success ? "success" : "failed",
                                            responseTime,
                                            reciept.BlockNumber.Value,
                                            reciept.TransactionHash));

                if (success)
                {
                    var devFee = (ulong)Math.Round(100 / Math.Abs(DevFee.UserPercent));
                    if (SubmittedShares == ulong.MaxValue) SubmittedShares = 0u;

                    if (((SubmittedShares) % devFee) == 0) SubmitDevFee(fromAddress, gasLimit, userGas, SubmittedShares);
                }
            }
            catch (AggregateException ex)
            {
                var errorMessage = "[ERROR] " + ex.Message;

                foreach (var iEx in ex.InnerExceptions)
                    errorMessage += "\n " + iEx.Message;

                Program.Print(errorMessage);
            }
            catch (Exception ex)
            {
                var errorMessage = "[ERROR] " + ex.Message;

                if (ex.InnerException != null)
                    errorMessage += "\n " + ex.InnerException.Message;

                Program.Print(errorMessage);
            }
        }

        private BigInteger GetMiningReward()
        {
            var failCount = 0;
            Program.Print("[INFO] Checking mining reward amount from network...");
            while (failCount < 10)
            {
                try
                {
                    return m_getMiningReward.CallAsync<BigInteger>().Result; // including decimals
                }
                catch (Exception) { failCount++; }
            }
            throw new Exception("Failed checking mining reward amount.");
        }

        private void SubmitDevFee(string fromAddress, HexBigInteger gasLimit, HexBigInteger userGas, ulong shareNo)
        {
            var success = false;
            var devTransactionID = string.Empty;
            TransactionReceipt devReciept = null;
            try
            {
                var miningReward = GetMiningReward();

                Program.Print(string.Format("[INFO] Transferring dev. fee for successful miner share [{0}]...", shareNo));

                var txInput = new object[] { DevFee.Address, miningReward };

                var txCount = m_web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(fromAddress).Result;

                var estimatedGasLimit = m_transferMethod.EstimateGasAsync(from: fromAddress,
                                                                          gas: gasLimit,
                                                                          value: new HexBigInteger(0),
                                                                          functionInput: txInput).Result;

                var transaction = m_transferMethod.CreateTransactionInput(from: fromAddress,
                                                                          gas: estimatedGasLimit,
                                                                          gasPrice: userGas,
                                                                          value: new HexBigInteger(0),
                                                                          functionInput: txInput);

                var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_account.PrivateKey,
                                                                              to: m_contract.Address,
                                                                              amount: 0,
                                                                              nonce: txCount.Value,
                                                                              gasPrice: userGas,
                                                                              gasLimit: estimatedGasLimit,
                                                                              data: transaction.Data);

                if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                    throw new Exception("Failed to verify transaction.");

                devTransactionID = m_web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;

                if (string.IsNullOrWhiteSpace(devTransactionID)) throw new Exception("Failed to submit dev fee.");

                while (devReciept == null)
                {
                    try
                    {
                        System.Threading.Thread.Sleep(3000);
                        devReciept = m_web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(devTransactionID).Result;
                    }
                    catch (AggregateException ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        foreach (var iEx in ex.InnerExceptions)
                            errorMessage += "\n " + iEx.Message;

                        Program.Print(errorMessage);
                    }
                    catch (Exception ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        if (ex.InnerException != null)
                            errorMessage += "\n " + ex.InnerException.Message;

                        Program.Print(errorMessage);
                    }
                }

                success = (devReciept.Status.Value == 1);

                if (!success) throw new Exception("Failed to submit dev fee.");
                else
                {
                    Program.Print(string.Format("[INFO] Transferred dev fee for successful mint share [{0}] : {1}, block: {2}," +
                                                "\n transaction ID: {3}",
                                                shareNo,
                                                success ? "success" : "failed",
                                                devReciept.BlockNumber.Value,
                                                devReciept.TransactionHash));
                }
            }
            catch (AggregateException ex)
            {
                var errorMessage = "[ERROR] " + ex.Message;

                foreach (var iEx in ex.InnerExceptions)
                    errorMessage += "\n " + iEx.Message;

                Program.Print(errorMessage);
            }
            catch (Exception ex)
            {
                var errorMessage = "[ERROR] " + ex.Message;

                if (ex.InnerException != null)
                    errorMessage += "\n " + ex.InnerException.Message;

                Program.Print(errorMessage);
            }
        }
    }
}