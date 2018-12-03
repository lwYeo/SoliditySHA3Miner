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

using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace SoliditySHA3Miner.API
{
    public class Ccminer
    {
        private const string ALGO = "soliditysha3";
        private const string EMULATE_API_VERSION = "1.9";
        private const string API_FORMAT = "Name={0};VER={1};API={2};ALGO={3};GPUS={4:D};KHS={5:F2};SOLV={6:D};ACC={7:D};REJ={8:D};ACCMN={9:F3};DIFF={10:F6};NETKHS={11:F0};POOLS={12:D};WAIT={13:D};UPTIME={14:F0};TS={15:D}|\r\n";

        private static Thread m_apiThread;
        private static Miner.IMiner[] m_miners;
        private static TcpListener m_currentListener;
        private static bool m_isRunning;

        private static uint GetUNIXCurrentTimestamp => (uint)(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1))).TotalSeconds;

        public static void StartListening(string apiBind, params Miner.IMiner[] miners)
        {
            if (m_apiThread != null
                && (m_apiThread.ThreadState != ThreadState.Aborted || m_apiThread.ThreadState != ThreadState.Stopped)) return;

            if (string.IsNullOrWhiteSpace(apiBind))
            {
                Program.Print("[INFO] minerCcminerAPI is null or empty, using default...");
                apiBind = Config.Defaults.CcminerAPIPath;
            }
            else if (apiBind == "0")
            {
                Program.Print("[INFO] ccminer-API is disabled.");
                return;
            }
            else if (!int.TryParse(apiBind.Split(':')[1], out int tempPort))
            {
                Program.Print("[ERROR] Invalid port provided for ccminer-API.");
                return;
            }
            else if (!IPAddress.TryParse(apiBind.Split(':')[0], out IPAddress tempIpAddress))
            {
                Program.Print("[ERROR] Invalid IP address provided for ccminer-API.");
                return;
            }

            var ipAddress = IPAddress.Parse(apiBind.Split(':')[0]);
            var port = (int)uint.Parse(apiBind.Split(':')[1]);

            try
            {
                var listener = new TcpListener(ipAddress, port);
                listener.Start();
                listener.Stop();
            }
            catch (Exception)
            {
                Program.Print("[ERROR] ccminer-API failed to bind to: " + apiBind);
                return;
            }

            m_miners = miners;

            m_apiThread = new Thread(() => Listen(ipAddress, port))
            {
                IsBackground = true
            };
            m_isRunning = true;
            m_apiThread.Start();
        }

        public static void StopListening()
        {
            if (m_apiThread != null)
            {
                Program.Print("[INFO] ccminer-API service stopping...");
                m_isRunning = false;
                try
                {
                    m_currentListener.Server.Close();
                    m_apiThread.Join(2000);
                }
                catch (Exception ex)
                {
                    Program.Print("[ERROR] An error has occured while stopping ccminer-API: " + ex.Message);
                }
            }
        }
        
        private static void Listen(IPAddress ipAddress, int port)
        {
            try
            {
                IPEndPoint localEndPoint = new IPEndPoint(ipAddress, port);
                m_currentListener = new TcpListener(localEndPoint);
                m_currentListener.Start();
                Program.Print(string.Format("[INFO] ccminer-API service started at {0}...", m_currentListener.LocalEndpoint));

                while (m_isRunning)
                {
                    try
                    {
                        var client = m_currentListener.AcceptTcpClient();

                        if (!m_isRunning) break;

                        using (var stream = client.GetStream())
                        {
                            var buffer = new byte[64];
                            var byteCount = stream.Read(buffer, 0, buffer.Length);
                            var request = Encoding.ASCII.GetString(buffer, 0, byteCount).
                                                         Replace('\r'.ToString(), string.Empty).
                                                         Replace('\n'.ToString(), string.Empty);

                            var response = string.Empty;
                            switch (request)
                            {
                                case "summary":
                                    var gpus = m_miners.SelectMany(m => m.Devices).Count(d => d.AllowDevice);
                                    var khs = m_miners.Sum((m => (long)m.GetTotalHashrate())) / 1000.0M;
                                    var solv = m_miners.Select(m => m.NetworkInterface).Distinct().Sum(i => (long)(i.SubmittedShares));
                                    var rej = m_miners.Select(m => m.NetworkInterface).Distinct().Sum(i => (long)(i.RejectedShares));
                                    var acc = solv - rej;
                                    var uptime = (DateTime.Now - Program.LaunchTime).TotalSeconds;
                                    var accmn = (60.0 * acc) / (uptime > 0.0 ? uptime : 1.0);
                                    double diff;
                                    try { diff = m_miners.Average(m => (long)m.NetworkInterface.Difficulty.Value); }
                                    catch { diff = long.MaxValue; }
                                    var netkhs = 0; // TODO: get network hashrate
                                    var pools = m_miners.Select(m => m.NetworkInterface).OfType<NetworkInterface.PoolInterface>().Distinct().Count();
                                    var wait = Program.WaitSeconds;
                                    var ts = GetUNIXCurrentTimestamp;

                                    response = string.Format(API_FORMAT,
                                                             Program.GetApplicationName(), Program.GetApplicationVersion(), EMULATE_API_VERSION, ALGO,
                                                             gpus, khs, solv, acc, rej, accmn, diff, netkhs, pools, wait, uptime, ts);
                                    break;
                            }
                            stream.Write(Encoding.ASCII.GetBytes(response), 0, response.Length);
                        }
                        client.Close();
                    }
                    catch (SocketException ex)
                    {
                        if (ex.ErrorCode != 10004)
                            Program.Print("[ERROR] " + ex.ToString());
                    }
                    catch (Exception ex) { Program.Print("[ERROR] " + ex.ToString()); }
                }
            }
            catch (Exception ex) { Program.Print("[ERROR] " + ex.ToString()); }
            Program.Print("[INFO] ccminer-API stopped.");
        }
    }
}