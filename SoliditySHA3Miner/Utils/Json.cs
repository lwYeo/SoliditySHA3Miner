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

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;

namespace SoliditySHA3Miner.Utils
{
    internal static class Json
    {
        private const int MAX_TIMEOUT = 10;
        private static object _Object1 = new object();
        private static object _Object2 = new object();
        private static object _Object3 = new object();
        private static object _Object4 = new object();
        private static object _Object5 = new object();
        private static object _Object6 = new object();
        private static object _Object7 = new object();

        public static readonly JsonSerializerSettings BaseClassFirstSettings =
            new JsonSerializerSettings { ContractResolver = BaseFirstContractResolver.Instance };

        public static string SerializeFromObject(object obj, JsonSerializerSettings settings = null)
        {
            lock (_Object1)
            {
                try
                {
                    return (settings == null) ?
                        JsonConvert.SerializeObject(obj, Formatting.Indented) :
                        JsonConvert.SerializeObject(obj, Formatting.Indented, settings);
                }
                catch { }
                return string.Empty;
            }
        }

        public static JObject DeserializeFromURL(string url)
        {
            lock (_Object2)
            {
                string sJSON = string.Empty;

                using (var oClient = new HttpClient())
                {
                    oClient.Timeout = new TimeSpan(0, 0, MAX_TIMEOUT);
                    using (HttpResponseMessage oResponse = oClient.GetAsync(url).Result)
                    {
                        using (HttpContent oContent = oResponse.Content)
                        {
                            sJSON = oContent.ReadAsStringAsync().Result;
                        }
                    }
                }
                return (JObject)JsonConvert.DeserializeObject(sJSON);
            }
        }

        public static T DeserializeFromURL<T>(string url)
        {
            lock (_Object3)
            {
                string sJSON = string.Empty;
                var jObject = (T)Activator.CreateInstance(typeof(T));
                try
                {
                    using (var oClient = new HttpClient())
                    {
                        oClient.Timeout = new TimeSpan(0, 0, MAX_TIMEOUT);
                        using (HttpResponseMessage oResponse = oClient.GetAsync(url).Result)
                        {
                            using (HttpContent oContent = oResponse.Content)
                            {
                                sJSON = oContent.ReadAsStringAsync().Result;
                            }
                        }
                    }
                    jObject = JsonConvert.DeserializeObject<T>(sJSON);
                }
                catch { }
                return jObject;
            }
        }

        public static T DeserializeFromFile<T>(string filePath)
        {
            lock(_Object6)
            {
                string sJSON = File.ReadAllText(filePath);
                T jObject = (T)Activator.CreateInstance(typeof(T));
                try
                {
                    jObject = JsonConvert.DeserializeObject<T>(sJSON);
                }
                catch { }
                return jObject;
            }
        }

        public static bool SerializeToFile(object jObject, string filePath, JsonSerializerSettings settings = null)
        {
            lock(_Object7)
            {
                try
                {
                    if (File.Exists(filePath)) File.Delete(filePath);
                    
                    File.WriteAllText(filePath, (settings == null) ?
                        JsonConvert.SerializeObject(jObject, Formatting.Indented) :
                        JsonConvert.SerializeObject(jObject, Formatting.Indented, settings));
                    return true;
                }
                catch
                {
                    return false;
                }
            }
        }

        public static JObject InvokeJObjectRPC(string url, JObject obj, JsonSerializerSettings settings = null, int customTimeout = 0)
        {
            lock (_Object5)
            {
                var serializedJSON = SerializeFromObject(obj, settings);

                var httpRequest = (HttpWebRequest)WebRequest.Create(url);
                httpRequest.Timeout = (customTimeout > 0 ? customTimeout : MAX_TIMEOUT) * 1000;
                httpRequest.ReadWriteTimeout = (customTimeout > 0 ? customTimeout : MAX_TIMEOUT) * 1000;
                httpRequest.ContentLength = serializedJSON.Length;
                httpRequest.ContentType = "application/json-rpc";
                httpRequest.Method = "POST";

                using (var streamWriter = new StreamWriter(httpRequest.GetRequestStream()))
                {
                    streamWriter.Write(serializedJSON);
                    streamWriter.Flush();
                    streamWriter.Close();
                }

                var httpResponse = (HttpWebResponse)httpRequest.GetResponse();
                using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
                {
                    var sResponse = streamReader.ReadToEnd();
                    var jResponse = JsonConvert.DeserializeObject<JObject>(sResponse);
                    return jResponse;
                }
            }
        }

        public static T CloneObject<T>(T objectToClone)
        {
            lock (_Object4)
            {
                if (objectToClone == null) { return default(T); }
                try
                {
                    return JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(objectToClone),
                                                            new JsonSerializerSettings() { ObjectCreationHandling = ObjectCreationHandling.Replace });
                }
                catch { return default(T); }
            }
        }

        public class ClassNameContractResolver : DefaultContractResolver
        {
            private ClassNameContractResolver()
            {
            }

            static ClassNameContractResolver() => Instance = new ClassNameContractResolver();

            public static ClassNameContractResolver Instance { get; private set; }

            protected override IList<JsonProperty> CreateProperties(Type type, MemberSerialization memberSerialization)
            {
                return base.CreateProperties(type, memberSerialization).
                            Select(p =>
                            {
                                p.PropertyName = p.UnderlyingName;
                                return p;
                            }).
                            ToList();
            }
        }

        public class BaseFirstContractResolver : DefaultContractResolver
        {
            private BaseFirstContractResolver()
            {
            }

            static BaseFirstContractResolver() => Instance = new BaseFirstContractResolver();

            public static BaseFirstContractResolver Instance { get; private set; }

            protected override IList<JsonProperty> CreateProperties(Type type, MemberSerialization memberSerialization)
            {
                return base.CreateProperties(type, memberSerialization).
                            OrderBy(p => p.DeclaringType.BaseTypesAndSelf().Count()).
                            ToList();
            }
        }
    }

    public static class TypeExtensions
    {
        public static IEnumerable<Type> BaseTypesAndSelf(this Type type)
        {
            while (type != null)
            {
                yield return type;
                type = type.BaseType;
            }
        }
    }
}