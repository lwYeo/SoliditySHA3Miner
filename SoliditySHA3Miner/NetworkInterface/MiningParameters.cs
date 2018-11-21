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

using Nethereum.Contracts;
using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using Newtonsoft.Json.Linq;
using System;
using System.Numerics;
using System.Threading.Tasks;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class MiningParameters
    {
        public HexBigInteger MiningDifficulty { get; private set; }
        public HexBigInteger MiningTarget { get; private set; }
        public HexBigInteger Challenge { get; private set; }
        public byte[] MiningTargetByte32 { get; private set; }
        public byte[] ChallengeByte32 { get; private set; }
        public string EthAddress { get; private set; }
        public string MiningTargetByte32String { get; private set; }
        public string ChallengeByte32String { get; private set; }

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

            while (retryCount < 10)
                try
                {
                    MiningDifficulty = new HexBigInteger(getMiningDifficulty.CallAsync<BigInteger>().Result);
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount < 10)
                        Task.Delay(200).Wait();
                    else
                        throw new OperationCanceledException("Failed to get difficulty from network: " + ex.Message, ex.InnerException);
                }

            while (retryCount < 10)
                try
                {
                    MiningTarget = new HexBigInteger(getMiningTarget.CallAsync<BigInteger>().Result);
                    MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false));
                    MiningTargetByte32String = Utils.Numerics.Byte32ArrayToHexString(MiningTargetByte32);
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount < 10)
                        Task.Delay(200).Wait();
                    else
                        throw new OperationCanceledException("Failed to get target from network: " + ex.Message, ex.InnerException);
                }

            while (retryCount < 10)
                try
                {
                    ChallengeByte32 = Utils.Numerics.FilterByte32Array(getChallengeNumber.CallAsync<byte[]>().Result);
                    Challenge = new HexBigInteger(HexByteConvertorExtensions.ToHex(ChallengeByte32, prefix: true));
                    ChallengeByte32String = Utils.Numerics.Byte32ArrayToHexString(ChallengeByte32);
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount < 10)
                        Task.Delay(200).Wait();
                    else
                        throw new OperationCanceledException("Failed to get challenge from network: " + ex.Message, ex.InnerException);
                }
        }

        private MiningParameters(string poolURL,
                                 JObject getPoolEthAddress,
                                 JObject getChallengeNumber,
                                 JObject getMinimumShareDifficulty,
                                 JObject getMinimumShareTarget)
        {
            var retryCount = 0;
            
            while (true)
                try
                {
                    EthAddress = Utils.Json.InvokeJObjectRPC(poolURL, getPoolEthAddress).SelectToken("$.result").Value<string>();
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount < 10)
                        Task.Delay(200).Wait();
                    else
                        throw new OperationCanceledException("Failed to get pool address: " + ex.Message, ex.InnerException);
                }

            while (true)
                try
                {
                    Challenge = new HexBigInteger(Utils.Json.InvokeJObjectRPC(poolURL, getChallengeNumber).SelectToken("$.result").Value<string>());
                    ChallengeByte32 = Utils.Numerics.FilterByte32Array(Challenge.Value.ToByteArray(littleEndian: false));
                    ChallengeByte32String = Utils.Numerics.Byte32ArrayToHexString(ChallengeByte32);
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount < 10)
                        Task.Delay(200).Wait();
                    else
                        throw new OperationCanceledException("Failed to get pool challenge: " + ex.Message, ex.InnerException);
                }

            while (true)
                try
                {
                    MiningDifficulty = new HexBigInteger(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareDifficulty).SelectToken("$.result").Value<ulong>());
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount < 10)
                        Task.Delay(200).Wait();
                    else
                        throw new OperationCanceledException("Failed to get pool difficulty: " + ex.Message, ex.InnerException);
                }

            while (true)
                try
                {
                    MiningTarget = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareTarget).SelectToken("$.result").Value<string>()));
                    MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false));
                    MiningTargetByte32String = Utils.Numerics.Byte32ArrayToHexString(MiningTargetByte32);
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount < 10)
                        Task.Delay(200).Wait();
                    else
                        throw new OperationCanceledException("Failed to get pool target: " + ex.Message, ex.InnerException);
                }
        }
    }
}