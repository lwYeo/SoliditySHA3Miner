using Nethereum.Contracts;
using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class MiningParameters
    {
        public HexBigInteger MiningDifficulty { get; private set; }
        public HexBigInteger MiningTarget { get; private set; }
        public HexBigInteger ChallengeNumber { get; private set; }
        public byte[] MiningTargetByte32 { get; private set; }
        public byte[] ChallengeNumberByte32 { get; private set; }
        public string EthAddress { get; private set; }
        public string MiningTargetByte32String { get; private set; }
        public string ChallengeNumberByte32String { get; private set; }

        public static MiningParameters GetSoloMiningParameters(string ethAddress,
                                                               Function getMiningDifficulty,
                                                               Function getMiningTarget,
                                                               Function getChallengeNumber)
        {
            return new MiningParameters(ethAddress, getMiningDifficulty, getMiningTarget, getChallengeNumber);
        }

        public static MiningParameters GetPoolMiningParameters(string poolURL,
                                                               JObject getPoolEthAddress,
                                                               JObject getChallengeNumber,
                                                               JObject getMinimumShareDifficulty,
                                                               JObject getMinimumShareTarget)
        {
            return new MiningParameters(poolURL, getPoolEthAddress, getChallengeNumber, getMinimumShareDifficulty, getMinimumShareTarget);
        }

        private MiningParameters(string ethAddress,
                                 Function getMiningDifficulty,
                                 Function getMiningTarget,
                                 Function getChallengeNumber)
        {
            EthAddress = ethAddress;

            var retryCount = 0;
            var exceptions = new List<Exception>();

            while (retryCount < 10)
            {
                try
                {
                    MiningDifficulty = new HexBigInteger(getMiningDifficulty.CallAsync<BigInteger>().Result);
                    break;
                }
                catch (AggregateException ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex.InnerExceptions[0]);
                    else { Task.Delay(200).Wait(); }
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex);
                    else { Task.Delay(200).Wait(); }
                }
            }

            while (retryCount < 10)
            {
                try
                {
                    MiningTarget = new HexBigInteger(getMiningTarget.CallAsync<BigInteger>().Result);
                    MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false));
                    MiningTargetByte32String = Utils.Numerics.BigIntegerToByte32HexString(MiningTarget.Value);
                    break;
                }
                catch (AggregateException ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex.InnerExceptions[0]);
                    else { Task.Delay(200).Wait(); }
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex);
                    else { Task.Delay(200).Wait(); }
                }
            }

            while (retryCount < 10)
            {
                try
                {
                    ChallengeNumberByte32 = Utils.Numerics.FilterByte32Array(getChallengeNumber.CallAsync<byte[]>().Result);
                    ChallengeNumber = new HexBigInteger(HexByteConvertorExtensions.ToHex(ChallengeNumberByte32, prefix: true));
                    ChallengeNumberByte32String = Utils.Numerics.BigIntegerToByte32HexString(ChallengeNumber.Value);
                    break;
                }
                catch (AggregateException ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex.InnerExceptions[0]);
                    else { Task.Delay(200).Wait(); }
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex);
                    else { Task.Delay(200).Wait(); }
                }
            }

            var exMessage = string.Join(Environment.NewLine, exceptions.Select(ex => ex.Message));
            if (exceptions.Any()) throw new Exception(exMessage);
        }

        private MiningParameters(string poolURL,
                                 JObject getPoolEthAddress,
                                 JObject getChallengeNumber,
                                 JObject getMinimumShareDifficulty,
                                 JObject getMinimumShareTarget)
        {
            EthAddress = Utils.Json.InvokeJObjectRPC(poolURL, getPoolEthAddress).SelectToken("$.result").Value<string>();
            ChallengeNumber = new HexBigInteger(Utils.Json.InvokeJObjectRPC(poolURL, getChallengeNumber).SelectToken("$.result").Value<string>());
            ChallengeNumberByte32 = Utils.Numerics.FilterByte32Array(ChallengeNumber.Value.ToByteArray(littleEndian: false));
            ChallengeNumberByte32String = Utils.Numerics.BigIntegerToByte32HexString(ChallengeNumber.Value);
            MiningDifficulty = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareDifficulty).SelectToken("$.result").Value<string>()));
            MiningTarget = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareTarget).SelectToken("$.result").Value<string>()));
            MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false));
            MiningTargetByte32String = Utils.Numerics.BigIntegerToByte32HexString(MiningTarget.Value);
        }
    }
}