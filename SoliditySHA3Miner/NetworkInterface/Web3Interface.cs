using System;
using System.Numerics;
using System.IO;
using Nethereum.Contracts;
using Nethereum.Hex.HexTypes;
using Nethereum.Util;
using Nethereum.Web3;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class Web3Interface : INetworkInterface
    {
        private const int MAX_TIMEOUT = 10;
        private const string DEFAULT_WEB3_API = "https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE";

        private readonly Web3 m_Web3;
        private readonly Contract m_Contract;
        private readonly string m_MinerAddress;
        private readonly string m_privateKey;

        public bool IsPool => false;
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }

        public long GetDifficulty()
        {
            return 0;
        }

        public Web3Interface(string web3ApiPath, string contractAddress, string minerAddress, string privateKey, string abiFileName)
        {
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
            if (!addressUtil.IsValidAddressLength(minerAddress))
            {
                throw new Exception("Invalid miner address provided.");
            }

            m_privateKey = privateKey;
            if (!string.IsNullOrWhiteSpace(m_privateKey))
            {
                //TODO: get miner address from private key
            }
            if (!string.IsNullOrWhiteSpace(minerAddress)) m_MinerAddress = minerAddress;

            Nethereum.JsonRpc.Client.ClientBase.ConnectionTimeout = MAX_TIMEOUT * 1000;

            m_Web3 = new Web3(string.IsNullOrWhiteSpace(web3ApiPath) ? DEFAULT_WEB3_API : web3ApiPath);

            var abi = File.ReadAllText(Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), abiFileName));

            m_Contract = m_Web3.Eth.GetContract(abi, contractAddress);
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
            

        }

        public void Dispose()
        {

        }
    }
}
