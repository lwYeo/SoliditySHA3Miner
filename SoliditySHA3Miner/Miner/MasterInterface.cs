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
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SoliditySHA3Miner.Miner.Device;
using SoliditySHA3Miner.NetworkInterface;
using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SoliditySHA3Miner.Miner
{
    public class MasterInterface : IMiner
    {
        #region static

        public static class RequestMethods
        {
            public const string GetMinerAddress = "GetMinerAddress";
            public const string GetMaximumTarget = "GetMaximumTarget";
            public const string GetKingAddress = "GetKingAddress";
            public const string GetChallenge = "GetChallenge";
            public const string GetDifficulty = "GetDifficulty";
            public const string GetTarget = "GetTarget";
            public const string GetPause = "GetPause";
            public const string GetPoolMining = "GetPoolMining";
            public const string SubmitSolution = "SubmitSolution";
        }

        public static JObject GetMasterParameter(string method, params string[] parameters)
        {
            var paramObject = new JObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = "1",
                ["method"] = method
            };
            if (parameters != null && parameters.Any())
            {
                var props = new JArray();
                foreach (var p in parameters)
                    props.Add(p);

                paramObject.Add(new JProperty("params", props));
            }
            return paramObject;
        }

        public static JObject GetMasterResult(params string[] results)
        {
            if (results.Length == 1)
            {
                return new JObject
                {
                    ["jsonrpc"] = "2.0",
                    ["id"] = "1",
                    ["result"] = results[0]
                };
            }
            else
            {
                var resultsArray = new JArray();
                foreach (var p in results)
                    resultsArray.Add(p);

                return new JObject
                {
                    ["jsonrpc"] = "2.0",
                    ["id"] = "1",
                    ["result"] = resultsArray
                };
            }
        }

        public static string GetMasterDefaultIpAddress()
        {
            var defaultIP = "127.0.0.1";
            string[] ipAddresses = { "1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4" };

            foreach (var address in ipAddresses)
                using (var socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.IP))
                    try
                    {
                        socket.Connect(address, 65530);
                        var endPoint = socket.LocalEndPoint as IPEndPoint;
                        defaultIP = endPoint.Address.ToString();
                        break;
                    }
                    catch { }

            return string.Format(Config.Defaults.MasterIpAddress, defaultIP);
        }

        #endregion

        public bool IsSupported { get; }
        public string IpAddress { get; }

        private readonly HttpListener m_Listener;
        private readonly int m_pauseOnFailedScan;
        private readonly string m_kingEthAddress;
        private readonly string m_maxTarget;
        private int m_failedScanCount;
        private bool m_isOngoing;
        private bool m_isCurrentChallengeStopSolving;
        private string m_minerEthAddress;
        private string m_challenge;
        private string m_difficulty;
        private string m_target;

        #region IMiner

        public INetworkInterface NetworkInterface { get; }
        public DeviceBase[] Devices => Enumerable.Empty<DeviceBase>().ToArray();
        public bool HasMonitoringAPI => false;
        public bool IsAnyInitialised => true;
        public bool IsMining => true;
        public bool IsStopped => false;
        public bool IsPause { get; private set; }
        public bool HasAssignedDevices { get; private set; }

        public void Dispose()
        {
            // Do nothing
        }

        public void StartMining(int networkUpdateInterval, int hashratePrintInterval)
        {
            // Do nothing
        }

        public void StopMining()
        {
            Program.Print("[INFO] Master instance stopping...");
            m_isOngoing = false;

            NetworkInterface.OnGetMiningParameterStatus -= NetworkInterface_OnGetMiningParameterStatus;
            NetworkInterface.OnNewChallenge -= NetworkInterface_OnNewChallenge;
            NetworkInterface.OnNewTarget -= NetworkInterface_OnNewTarget;
            NetworkInterface.OnNewDifficulty -= NetworkInterface_OnNewDifficulty;
            NetworkInterface.OnStopSolvingCurrentChallenge -= NetworkInterface_OnStopSolvingCurrentChallenge;

            m_Listener?.Stop();
            m_Listener?.Close();
        }

        public ulong GetTotalHashrate()
        {
            return 0ul;
        }

        public ulong GetHashRateByDevice(DeviceBase device)
        {
            return 0ul;
        }

        #endregion

        public MasterInterface(INetworkInterface networkInterface, int pauseOnFailedScans, string ipAddress = null)
        {
            m_failedScanCount = 0;
            m_pauseOnFailedScan = pauseOnFailedScans;
            NetworkInterface = networkInterface;
            NetworkInterface.OnGetMiningParameterStatus += NetworkInterface_OnGetMiningParameterStatus;
            NetworkInterface.OnNewChallenge += NetworkInterface_OnNewChallenge;
            NetworkInterface.OnNewTarget += NetworkInterface_OnNewTarget;
            NetworkInterface.OnNewDifficulty += NetworkInterface_OnNewDifficulty;
            NetworkInterface.OnStopSolvingCurrentChallenge += NetworkInterface_OnStopSolvingCurrentChallenge;

            IsSupported = HttpListener.IsSupported;
            if (!IsSupported)
            {
                Program.Print("[ERROR] Obsolete OS detected, Master instance will not start.");
                return;
            }

            if (string.IsNullOrWhiteSpace(ipAddress))
            {
                ipAddress = GetMasterDefaultIpAddress();
                Program.Print(string.Format("[INFO] masterIpAddress is null or empty, using default {0}", ipAddress));
            }

            if (!ipAddress.StartsWith("http://") || ipAddress.StartsWith("https://"))
                ipAddress = "http://" + ipAddress;

            if (!ipAddress.EndsWith("/")) ipAddress += "/";

            if (!int.TryParse(ipAddress.Split(':')[2].TrimEnd('/'), out int port))
            {
                Program.Print("[ERROR] Invalid port provided for masterIpAddress.");
                return;
            }

            var tempIPAddress = ipAddress.Split(new string[] { "//" }, StringSplitOptions.None)[1].Split(':')[0];
            if (!IPAddress.TryParse(tempIPAddress, out IPAddress ipAddressObj))
            {
                Program.Print("[ERROR] Invalid IP address provided for Master instance.");
                return;
            }

            using (var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
            {
                try { socket.Bind(new IPEndPoint(ipAddressObj, port)); }
                catch (Exception)
                {
                    Program.Print("[ERROR] Master instance failed to bind to: " + ipAddress);
                    return;
                }
            };

            try
            {
                if (Work.KingAddress != null)
                    m_kingEthAddress = Utils.Numerics.Byte20ArrayToAddressString(Work.KingAddress);

                m_maxTarget = Utils.Numerics.Byte32ArrayToHexString(
                    networkInterface.MaxTarget.Value.ToByteArray(isUnsigned: true, isBigEndian: true));

                if (networkInterface.CurrentChallenge == null)
                    networkInterface.UpdateMiningParameters();
                else
                {
                    m_minerEthAddress = networkInterface.MinerAddress;
                    m_challenge = Utils.Numerics.Byte32ArrayToHexString(networkInterface.CurrentChallenge);
                    m_difficulty = Utils.Numerics.Byte32ArrayToHexString(
                        networkInterface.Difficulty.Value.ToByteArray(isUnsigned: true, isBigEndian: true));
                    m_target = Utils.Numerics.Byte32ArrayToHexString(
                        networkInterface.CurrentTarget.Value.ToByteArray(isUnsigned: true, isBigEndian: true));
                }

                m_Listener = new HttpListener();
                m_Listener.Prefixes.Add(ipAddress);
                m_Listener.Start();

                Process(m_Listener);
            }
            catch (HttpListenerException ex)
            {
                HandleException(ex, errorPostfix: string.Format("Listening failed at ({0}): Check URL validity, firewall settings, and admin/sudo mode.", ipAddress));
                Environment.Exit(1);
            }
            catch (Exception ex)
            {
                HandleException(ex);
                return;
            }
        }

        private async void Process(HttpListener listener)
        {
            HasAssignedDevices = true;
            Program.Print(string.Format("[INFO] Master instance started at {0}...", listener.Prefixes.ElementAt(0)));

            m_isOngoing = true;
            while (m_isOngoing)
            {
                HttpListenerContext context = null;

                try { context = await listener.GetContextAsync(); }
                catch (HttpListenerException ex)
                {
                    if (ex.ErrorCode == 995) break;

                    HandleException(ex, string.Format("{0} Error code: {1}, Message: {2}",
                                                      ex.GetType().Name, ex.ErrorCode, ex.Message));
                    await Task.Delay(1000);
                    continue;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    HandleException(ex);
                    await Task.Delay(1000);
                    continue;
                }

                var request = context.Request;
                var response = context.Response;
                if (request != null && response != null)
                {
                    response.AppendHeader("Pragma", "no-cache");
                    response.AppendHeader("Expires", "0");
                    response.ContentType = "application/json";
                    response.StatusCode = (int)HttpStatusCode.OK;

                    ProcessApiDataResponse(request, response);
                }
            }
        }

        private void ProcessApiDataResponse(HttpListenerRequest request, HttpListenerResponse response)
        {
            if (response != null)
                Task.Factory.StartNew(() =>
                {
                    try
                    {
                        var requestJSON = string.Empty;
                        JObject jResponse = null;

                        if (request.HasEntityBody)
                            using (var body = request.InputStream)
                            using (var reader = new System.IO.StreamReader(body, request.ContentEncoding))
                                requestJSON = reader.ReadToEnd();
                        try
                        {
                            var jRequest = (JObject)JsonConvert.DeserializeObject(requestJSON);
                            var jMethodName = jRequest.SelectToken("$.method").Value<string>();

                            switch (jMethodName)
                            {
                                case RequestMethods.GetMinerAddress:
                                    jResponse = GetMasterResult(m_minerEthAddress);
                                    break;

                                case RequestMethods.GetMaximumTarget:
                                    jResponse = GetMasterResult(m_maxTarget);
                                    break;

                                case RequestMethods.GetKingAddress:
                                    jResponse = GetMasterResult(m_kingEthAddress);
                                    break;

                                case RequestMethods.GetChallenge:
                                    jResponse = GetMasterResult(m_challenge);
                                    break;

                                case RequestMethods.GetDifficulty:
                                    jResponse = GetMasterResult(m_difficulty);
                                    break;

                                case RequestMethods.GetTarget:
                                    jResponse = GetMasterResult(m_target);
                                    break;

                                case RequestMethods.GetPause:
                                    jResponse = GetMasterResult(IsPause.ToString());
                                    break;

                                case RequestMethods.GetPoolMining:
                                    jResponse = GetMasterResult(NetworkInterface.IsPool.ToString());
                                    break;

                                case RequestMethods.SubmitSolution:
                                    var slaveURL = request.RemoteEndPoint.ToString();
                                    Program.Print("[INFO] Solution received from slave URL: http://" + slaveURL);

                                    var jParams = jRequest.SelectToken("$.params").Value<JArray>();
                                    var digest = Utils.Numerics.HexStringToByte32Array(jParams[0].Value<string>());
                                    var challenge = Utils.Numerics.HexStringToByte32Array(jParams[1].Value<string>());
                                    var difficulty = Utils.Numerics.HexStringToByte32Array(jParams[2].Value<string>());
                                    var nonce = Utils.Numerics.HexStringToByte32Array(jParams[3].Value<string>());
                                    var difficultyBigInteger = new HexBigInteger(new BigInteger(difficulty, isUnsigned: true, isBigEndian: true));

                                    var result = NetworkInterface.SubmitSolution(m_minerEthAddress, digest, challenge, difficultyBigInteger, nonce, this);
                                    jResponse = GetMasterResult(result.ToString());
                                    break;
                            }
                        }
                        catch (Exception ex)
                        {
                            HandleException(ex);
                        }

                        var buffer = Encoding.UTF8.GetBytes(Utils.Json.SerializeFromObject(jResponse));
                        if (buffer != null)
                            using (var output = response.OutputStream)
                            {
                                output.Write(buffer, 0, buffer.Length);
                                output.Flush();
                            }
                    }
                    catch (Exception ex)
                    {
                        response.StatusCode = (int)HttpStatusCode.InternalServerError;
                        HandleException(ex);
                    }
                    finally
                    {
                        try { response.Close(); }
                        catch (Exception ex) { HandleException(ex); }
                    }
                },
            TaskCreationOptions.LongRunning);
        }

        private void NetworkInterface_OnStopSolvingCurrentChallenge(INetworkInterface sender, bool stopSolving = true)
        {
            m_isCurrentChallengeStopSolving = true;
            IsPause = true;
        }

        private void NetworkInterface_OnNewTarget(INetworkInterface sender, HexBigInteger target)
        {
            m_target = Utils.Numerics.Byte32ArrayToHexString(target.Value.ToByteArray(isUnsigned: true, isBigEndian: true));
        }

        private void NetworkInterface_OnNewDifficulty(INetworkInterface sender, HexBigInteger difficulty)
        {
            m_difficulty = Utils.Numerics.Byte32ArrayToHexString(difficulty.Value.ToByteArray(isUnsigned: true, isBigEndian: true));
        }

        private void NetworkInterface_OnNewChallenge(INetworkInterface sender, byte[] challenge, string address)
        {
            m_challenge = Utils.Numerics.Byte32ArrayToHexString(challenge);
            m_minerEthAddress = address;

            if (m_isCurrentChallengeStopSolving)
            {
                IsPause = false;
                m_isCurrentChallengeStopSolving = false;
            }
        }

        private void NetworkInterface_OnGetMiningParameterStatus(INetworkInterface sender, bool success)
        {
            if (success)
            {
                if (m_isCurrentChallengeStopSolving)
                    IsPause = true;

                else if (IsPause)
                {
                    if (m_failedScanCount > m_pauseOnFailedScan)
                        m_failedScanCount = 0;

                    IsPause = false;
                }
            }
            else
            {
                m_failedScanCount++;
                
                if (m_failedScanCount > m_pauseOnFailedScan)
                    IsPause = true;
            }
        }

        private void HandleException(Exception ex, string errorPrefix = null, string errorPostfix = null)
        {
            var errorMessage = new StringBuilder("[ERROR] Occured at Master instance => ");

            if (!string.IsNullOrWhiteSpace(errorPrefix))
                errorMessage.AppendFormat("{0}: ", errorPrefix);

            errorMessage.Append(ex.Message);

            var innerEx = ex.InnerException;
            while (innerEx != null)
            {
                errorMessage.AppendFormat("\n {0}", innerEx.Message);
                innerEx = innerEx.InnerException;
            }

            if (!string.IsNullOrWhiteSpace(errorPostfix))
                errorMessage.AppendFormat("\n => {0}", errorPostfix);

            Program.Print(errorMessage.ToString());
        }
    }
}