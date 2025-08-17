using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;

namespace AI生成AI
{
    public class StructureValidator
    {
        private const string DeepSeekApiUrl = "https://api.deepseek.com/v1/chat/completions";
        private string apiKey = "sk-e4d234ec0ce5425cb84a54bd227f32f7";

        public async Task<(int operationsApplied, string result)> ValidateAndFixStructure(string projectRoot)
        {
            string structureDescription = BuildStructureDescription(projectRoot);
            string aiResponse = await SendToDeepSeek(structureDescription);
            return ApplyStructureFix(projectRoot, aiResponse);
        }

        private string BuildStructureDescription(string rootPath)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("项目结构树:");
            BuildStructureRecursive(rootPath, "", sb);
            return sb.ToString();
        }

        private void BuildStructureRecursive(string path, string indent, StringBuilder sb)
        {
            try
            {
                // 添加当前目录
                sb.AppendLine($"{indent}├─ {Path.GetFileName(path)}/");

                // 添加文件
                foreach (string file in Directory.GetFiles(path))
                {
                    sb.AppendLine($"{indent}│  ├─ {Path.GetFileName(file)}");
                }

                // 递归添加子目录
                foreach (string dir in Directory.GetDirectories(path))
                {
                    BuildStructureRecursive(dir, indent + "│  ", sb);
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine($"{indent}│  [错误: {ex.Message}]");
            }
        }

        private async Task<string> SendToDeepSeek(string structureDescription)
        {
            using var client = new HttpClient();
            client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

            var requestData = new
            {
                model = "deepseek-coder",
                messages = new[]
                {
                    new {
                        role = "system",
                        content = "你是一个AI项目结构修复专家。用户会提供项目结构树，你需要检测并修复结构问题：" +
                                  "1. 修复缺失的关键文件和文件夹；" +
                                  "2. 修复错误的文件位置；" +
                                  "3. 添加必要的目录结构；" +
                                  "4. 调整结构顺序（按功能模块分组）；" +
                                  "5. 整理用户添加的新结构到正确位置；" +
                                  "6. 返回JSON格式的修复操作列表：{\"operations\": [{\"type\": \"create_folder|create_file|move_file|reorder\", \"path\": \"完整路径\", \"content\": \"文件内容(如果是文件)\", \"order\": [\"文件1\", \"文件2\", \"文件夹1\"]}]}\n" +
                                  "reorder操作用于调整顺序，order属性包含该目录下所有项目的新顺序列表"
                    },
                    new {
                        role = "user",
                        content = $"项目结构树:\n{structureDescription}\n\n请修复以上结构问题并调整顺序"
                    }
                },
                temperature = 0.1,
                max_tokens = 4000
            };

            var requestContent = new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json");
            var response = await client.PostAsync(DeepSeekApiUrl, requestContent);

            if (!response.IsSuccessStatusCode)
            {
                string errorContent = await response.Content.ReadAsStringAsync();
                throw new Exception($"API错误: {response.StatusCode} - {errorContent}");
            }

            return await response.Content.ReadAsStringAsync();
        }

        public (int operationsApplied, string result) ApplyStructureFix(string projectRoot, string apiResponse)
        {
            int operationsApplied = 0;
            StringBuilder resultLog = new StringBuilder();

            try
            {
                // 解析API响应
                JObject responseObj = JObject.Parse(apiResponse);
                JToken choicesToken = responseObj["choices"];

                if (choicesToken == null || !choicesToken.Any())
                {
                    return (0, "API响应中缺少choices字段");
                }

                JToken messageToken = choicesToken[0]?["message"];
                if (messageToken == null)
                {
                    return (0, "API响应中缺少message字段");
                }

                string content = messageToken["content"]?.ToString();
                if (string.IsNullOrWhiteSpace(content))
                {
                    return (0, "API响应中content为空");
                }

                // 尝试解析JSON内容
                JObject result;
                try
                {
                    result = JObject.Parse(content);
                }
                catch
                {
                    // 如果解析失败，尝试提取JSON块
                    var jsonMatch = Regex.Match(content, @"```json\s*(\{.*\})\s*```", RegexOptions.Singleline);
                    if (jsonMatch.Success)
                    {
                        result = JObject.Parse(jsonMatch.Groups[1].Value);
                    }
                    else
                    {
                        return (0, "无法从响应中提取JSON数据");
                    }
                }

                JArray operations = result["operations"] as JArray;

                if (operations == null || operations.Count == 0)
                {
                    return (0, "未找到修复操作");
                }

                // 先执行所有非reorder操作
                foreach (var operation in operations)
                {
                    string type = operation["type"]?.ToString();
                    string path = operation["path"]?.ToString();
                    string fileContent = operation["content"]?.ToString();

                    if (string.IsNullOrWhiteSpace(type) || string.IsNullOrWhiteSpace(path))
                        continue;

                    if (type == "reorder") continue; // 稍后处理

                    try
                    {
                        string fullPath = Path.Combine(projectRoot, path);

                        switch (type)
                        {
                            case "create_folder":
                                if (!Directory.Exists(fullPath))
                                {
                                    Directory.CreateDirectory(fullPath);
                                    resultLog.AppendLine($"已创建文件夹: {path}");
                                    operationsApplied++;
                                }
                                break;

                            case "create_file":
                                if (!File.Exists(fullPath))
                                {
                                    Directory.CreateDirectory(Path.GetDirectoryName(fullPath));
                                    File.WriteAllText(fullPath, fileContent ?? "");
                                    resultLog.AppendLine($"已创建文件: {path}");
                                    operationsApplied++;
                                }
                                break;

                            case "move_file":
                                string source = operation["source"]?.ToString();
                                if (!string.IsNullOrWhiteSpace(source))
                                {
                                    string fullSource = Path.Combine(projectRoot, source);
                                    if (File.Exists(fullSource))
                                    {
                                        Directory.CreateDirectory(Path.GetDirectoryName(fullPath));
                                        File.Move(fullSource, fullPath);
                                        resultLog.AppendLine($"已移动文件: {source} -> {path}");
                                        operationsApplied++;
                                    }
                                }
                                break;
                        }
                    }
                    catch (Exception ex)
                    {
                        resultLog.AppendLine($"操作失败({type} {path}): {ex.Message}");
                    }
                }

                // 再执行reorder操作
                foreach (var operation in operations)
                {
                    string type = operation["type"]?.ToString();
                    if (type != "reorder") continue;

                    string path = operation["path"]?.ToString();
                    JArray orderArray = operation["order"] as JArray;

                    if (string.IsNullOrWhiteSpace(path) || orderArray == null || orderArray.Count == 0)
                        continue;

                    try
                    {
                        string fullPath = Path.Combine(projectRoot, path);
                        if (Directory.Exists(fullPath))
                        {
                            string tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
                            Directory.CreateDirectory(tempDir);

                            // 获取目录下所有项目
                            var items = Directory.GetFileSystemEntries(fullPath).ToList();
                            var newOrder = orderArray.Select(x => x.ToString()).ToList();

                            // 将项目移动到临时目录
                            foreach (var item in items)
                            {
                                string tempPath = Path.Combine(tempDir, Path.GetFileName(item));
                                if (File.Exists(item))
                                {
                                    File.Move(item, tempPath);
                                }
                                else if (Directory.Exists(item))
                                {
                                    Directory.Move(item, tempPath);
                                }
                            }

                            // 按新顺序移回项目
                            foreach (var itemName in newOrder)
                            {
                                string sourcePath = Path.Combine(tempDir, itemName);
                                string destPath = Path.Combine(fullPath, itemName);

                                if (File.Exists(sourcePath))
                                {
                                    File.Move(sourcePath, destPath);
                                }
                                else if (Directory.Exists(sourcePath))
                                {
                                    Directory.Move(sourcePath, destPath);
                                }
                            }

                            // 删除临时目录
                            Directory.Delete(tempDir, true);

                            resultLog.AppendLine($"已调整顺序: {path}");
                            operationsApplied++;
                        }
                    }
                    catch (Exception ex)
                    {
                        resultLog.AppendLine($"调整顺序失败({path}): {ex.Message}");
                    }
                }

                return operationsApplied > 0
                    ? (operationsApplied, $"已应用 {operationsApplied} 个修复操作:\n{resultLog}")
                    : (0, "未应用任何修复操作");
            }
            catch (Exception ex)
            {
                return (0, $"修复结构失败: {ex.Message}\n原始响应:\n{apiResponse}");
            }
        }
    }
}