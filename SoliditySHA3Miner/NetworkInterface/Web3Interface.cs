using System;
using System.Numerics;
using System.IO;
using Nethereum.Contracts;
using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using Nethereum.RPC.Eth.DTOs;
using Nethereum.Util;
using Nethereum.Web3;
using Nethereum.Web3.Accounts;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class Web3Interface : INetworkInterface
    {
        private const int MAX_TIMEOUT = 10;
        private const string DEFAULT_WEB3_API = "https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE";

        private readonly Web3 m_Web3;
        private readonly Contract m_Contract;
        private readonly Account m_Account;
        private readonly float m_GasToMine;
        private readonly string m_MinerAddress;
        private readonly BigInteger m_MiningReward;
        private readonly Function m_MintMethod;
        private readonly Function m_TransferMethod;

        public bool IsPool => false;
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }

        public long GetDifficulty()
        {
            return 0;
        }

        public Web3Interface(string web3ApiPath, string contractAddress, string minerAddress, string privateKey, float gasToMine, string abiFileName)
        {
            Nethereum.JsonRpc.Client.ClientBase.ConnectionTimeout = MAX_TIMEOUT * 1000;

            if (string.IsNullOrWhiteSpace(contractAddress))
            {
                Program.Print("[INFO] Contract address not specified, default 0xBTC");
                contractAddress = "0xB6eD7644C69416d67B522e20bC294A9a9B405B31";
            }

            var addressUtil = new AddressUtil();
            if (!addressUtil.IsValidAddressLength(contractAddress) || !addressUtil.IsChecksumAddress(contractAddress))
            {
                throw new Exception("Invalid contract address provided.");
            }

            if (!string.IsNullOrWhiteSpace(privateKey))
            {
                m_Account = new Account(privateKey);
                minerAddress = m_Account.Address;
            }

            if (!addressUtil.IsValidAddressLength(minerAddress))
            {
                throw new Exception("Invalid miner address provided.");
            }

            m_MinerAddress = minerAddress;

            m_Web3 = new Web3(string.IsNullOrWhiteSpace(web3ApiPath) ? DEFAULT_WEB3_API : web3ApiPath);

            var abi = File.ReadAllText(Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), abiFileName));

            m_Contract = m_Web3.Eth.GetContract(abi, contractAddress);

            if (!string.IsNullOrWhiteSpace(privateKey))
            {
                m_MintMethod = m_Contract.GetFunction("mint");

                m_TransferMethod = m_Contract.GetFunction("transfer");

                m_MiningReward = GetMiningReward();

                m_GasToMine = gasToMine;
            }
        }

        public HexBigInteger GetMaxDifficulity()
        {
            Program.Print("[INFO] Getting maximum difficulity from network...");
            while (true)
            {
                try
                {
                    return new HexBigInteger(m_Contract.GetFunction("_MAXIMUM_TARGET").CallAsync<BigInteger>().Result);
                }
                catch (AggregateException ex)
                {
                    Program.Print("[ERROR] " + ex.InnerExceptions[0].Message);
                }
            }
        }

        public MiningParameters GetMiningParameters()
        {
            return MiningParameters.GetSoloMiningParameters(m_Contract, m_MinerAddress);
        }

        void INetworkInterface.SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, bool isCustomDifficulty)
        {
            lock(this)
            {
                var success = false;
                var transactionID = string.Empty;
                var gasLimit = new HexBigInteger(1704624ul);
                var userGas = new HexBigInteger(UnitConversion.Convert.ToWei(new BigDecimal(m_GasToMine), UnitConversion.EthUnit.Gwei));
                var dataInput = new object[] { new HexBigInteger(solution).Value, HexByteConvertorExtensions.HexToByteArray(digest) };

                while (string.IsNullOrWhiteSpace(transactionID))
                {
                    try
                    {
                        System.Threading.Thread.Sleep(1000);

                        var txCount = m_Web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(fromAddress).Result;

                        var estimatedGasLimit = m_MintMethod.EstimateGasAsync(from: fromAddress,
                                                                              gas: gasLimit,
                                                                              value: new HexBigInteger(0),
                                                                              functionInput: dataInput).Result;

                        var transaction = m_MintMethod.CreateTransactionInput(from: fromAddress,
                                                                              gas: estimatedGasLimit,
                                                                              gasPrice: userGas,
                                                                              value: new HexBigInteger(0),
                                                                              functionInput: dataInput);

                        var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_Account.PrivateKey,
                                                                                      to: m_Contract.Address,
                                                                                      amount: 0,
                                                                                      nonce: txCount.Value,
                                                                                      gasPrice: userGas,
                                                                                      gasLimit: estimatedGasLimit,
                                                                                      data: transaction.Data);

                        if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                            throw new Exception("Failed to verify transaction.");

                        transactionID = m_Web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;

                        if (string.IsNullOrWhiteSpace(transactionID)) continue;
                        else
                        {
                            TransactionReceipt reciept = null;
                            while (reciept == null)
                            {
                                try
                                {
                                    System.Threading.Thread.Sleep(3000);
                                    reciept = m_Web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(transactionID).Result;
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
                            SubmittedShares++;

                            Program.Print(string.Format("[INFO] Share [{0}] submitted: {1}, block: {2}," +
                                                        "\n transaction ID: {3}",
                                                        SubmittedShares,
                                                        success ? "success" : "failed",
                                                        reciept.BlockNumber.Value,
                                                        reciept.TransactionHash));
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
                        if (ex.Message == "Failed to verify transaction.") break;
                    }
                }

                if (success)
                {
                    var devTransactionID = string.Empty;
                    TransactionReceipt devReciept = null;
                    var donate = (ulong)Math.Round(100 / Math.Abs(Donation.UserPercent));
                    if (SubmittedShares == ulong.MaxValue) SubmittedShares = 0u;

                    if (((SubmittedShares) % donate) == 0)
                    {
                        try
                        {
                            Program.Print(string.Format("[INFO] Transferring donation fee for share [{0}]...", SubmittedShares));

                            var txInput = new object[] { Donation.Address, m_MiningReward };

                            var txCount = m_Web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(fromAddress).Result;

                            var estimatedGasLimit = m_TransferMethod.EstimateGasAsync(from: fromAddress,
                                                                                      gas: gasLimit,
                                                                                      value: new HexBigInteger(0),
                                                                                      functionInput: txInput).Result;

                            var transaction = m_TransferMethod.CreateTransactionInput(from: fromAddress,
                                                                                      gas: estimatedGasLimit,
                                                                                      gasPrice: userGas,
                                                                                      value: new HexBigInteger(0),
                                                                                      functionInput: txInput);

                            var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_Account.PrivateKey,
                                                                                          to: m_Contract.Address,
                                                                                          amount: 0,
                                                                                          nonce: txCount.Value,
                                                                                          gasPrice: userGas,
                                                                                          gasLimit: estimatedGasLimit,
                                                                                          data: transaction.Data);

                            if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                                throw new Exception("Failed to verify transaction.");

                            devTransactionID = m_Web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;

                            if (string.IsNullOrWhiteSpace(devTransactionID)) throw new Exception("Failed to submit donation fee.");

                            while (devReciept == null)
                            {
                                try
                                {
                                    System.Threading.Thread.Sleep(3000);
                                    devReciept = m_Web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(devTransactionID).Result;
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

                            if (!success) throw new Exception("Failed to submit donation fee.");
                            else
                            {
                                Program.Print(string.Format("[INFO] Transferred donation fee for share [{0}] : {1}, block: {2}," +
                                                            "\n transaction ID: {3}",
                                                            SubmittedShares,
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
        }

        public void Dispose()
        {

        }

        private BigInteger GetMiningReward()
        {
            var failCount = 0;
            Program.Print("[INFO] Getting mining reward amount from network...");
            while (failCount < 10)
            {
                try
                {
                    return m_Contract.GetFunction("getMiningReward").CallAsync<BigInteger>().Result; // including decimals
                }
                catch (Exception) { failCount++; }
            }
            throw new Exception("Failed getting mining reward amount.");
        }
    }
}
