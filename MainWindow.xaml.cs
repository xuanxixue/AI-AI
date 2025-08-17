using ICSharpCode.AvalonEdit;
using ICSharpCode.AvalonEdit.Highlighting;
using Microsoft.Win32;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using SharpCompress.Archives;
using WinForms = System.Windows.Forms;

namespace AI生成AI
{
    public class BoolToFontWeightConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
            => (value is bool and true) ? FontWeights.Bold : FontWeights.Normal;

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
            => throw new NotImplementedException();
    }

    public class ResolvedStatusConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
            => (value is bool resolved) ? (resolved ? "已解决" : "未解决") : "未知";

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
            => throw new NotImplementedException();
    }

    public partial class MainWindow : Window
    {
        public ObservableCollection<ErrorItem> Errors { get; } = new ObservableCollection<ErrorItem>();
        private Dictionary<string, TabItem> openTabs = new Dictionary<string, TabItem>();
        private Dictionary<string, string> originalCodeCache = new Dictionary<string, string>();
        private const string DeepSeekApiUrl = "https://api.deepseek.com/v1/chat/completions";
        private bool isProcessingAIRequest = false;
        private bool isTraining = false;
        private CancellationTokenSource trainingCancellationTokenSource;
        public string currentProjectRoot = "";
        private StructureValidator structureValidator = new StructureValidator();

        public MainWindow()
        {
            InitializeComponent();
            SetupSyntaxHighlighting();
            lstErrors.ItemsSource = Errors;
            ConsoleWriteLine("OLMo Manager 已启动");
            txtTrainingLog.Text = "训练日志就绪...\n";
            UpdateWeightPreview();

            // 注册单选按钮事件
            rbImportZip.Checked += RbImportZip_Checked;
            rbImportFolder.Checked += RbImportFolder_Checked;
            rbImportFile.Checked += RbImportFile_Checked;
        }

        private void SetupSyntaxHighlighting()
        {
            HighlightingManager.Instance.RegisterHighlighting("Python", new[] { ".py" }, HighlightingManager.Instance.GetDefinition("Python"));
            HighlightingManager.Instance.RegisterHighlighting("JSON", new[] { ".json" }, HighlightingManager.Instance.GetDefinition("JavaScript"));
        }

        private void btnImport_Click(object sender, RoutedEventArgs e)
        {
            importPanel.Visibility = importPanel.Visibility == Visibility.Visible ? Visibility.Collapsed : Visibility.Visible;
            structureTreePanel.Visibility = Visibility.Collapsed;
        }

        private void BrowseZip_Click(object sender, RoutedEventArgs e)
        {
            FileImporter.BrowseZip(txtZipPath);
        }

        private void BrowseFolder_Click(object sender, RoutedEventArgs e)
        {
            FileImporter.BrowseFolder(txtExtractPath);
        }

        private void Extract_Click(object sender, RoutedEventArgs e)
        {
            FileImporter.ExtractZip(txtZipPath.Text, txtExtractPath.Text, txtLog);
            if (!string.IsNullOrWhiteSpace(txtExtractPath.Text))
            {
                LoadFileStructure(txtExtractPath.Text);
                currentProjectRoot = txtExtractPath.Text;
            }
        }

        private void RbImportZip_Checked(object sender, RoutedEventArgs e)
        {
            zipImportPanel.Visibility = Visibility.Visible;
            folderImportPanel.Visibility = Visibility.Collapsed;
            fileImportPanel.Visibility = Visibility.Collapsed;
        }

        private void RbImportFolder_Checked(object sender, RoutedEventArgs e)
        {
            zipImportPanel.Visibility = Visibility.Collapsed;
            folderImportPanel.Visibility = Visibility.Visible;
            fileImportPanel.Visibility = Visibility.Collapsed;
        }

        private void RbImportFile_Checked(object sender, RoutedEventArgs e)
        {
            zipImportPanel.Visibility = Visibility.Collapsed;
            folderImportPanel.Visibility = Visibility.Collapsed;
            fileImportPanel.Visibility = Visibility.Visible;
        }

        private void BrowseFolderImport_Click(object sender, RoutedEventArgs e)
        {
            FileImporter.BrowseFolderImport(txtFolderPath);
        }

        private void ImportFolder_Click(object sender, RoutedEventArgs e)
        {
            FileImporter.ImportFolder(txtFolderPath.Text, fileTreeView, txtLog, ref currentProjectRoot);
        }

        private void BrowseFileImport_Click(object sender, RoutedEventArgs e)
        {
            FileImporter.BrowseFileImport(txtFilePath);
        }

        private void ImportFile_Click(object sender, RoutedEventArgs e)
        {
            FileImporter.ImportFile(txtFilePath.Text, fileTreeView, txtLog, ref currentProjectRoot);
        }

        private void LoadFileStructure(string rootPath)
        {
            fileTreeView.Items.Clear();
            openTabs.Clear();
            tabControl.Items.Clear();
            originalCodeCache.Clear();

            if (!Directory.Exists(rootPath)) return;

            var rootNode = new FileNode(Path.GetFileName(rootPath), rootPath, true);
            BuildTree(rootNode, rootPath);
            fileTreeView.Items.Add(rootNode);

            Dispatcher.BeginInvoke(() =>
            {
                if (fileTreeView.ItemContainerGenerator.ContainerFromItem(rootNode) is TreeViewItem item)
                {
                    item.IsExpanded = true;
                    ExpandAllChildren(item);
                }
            }, DispatcherPriority.ContextIdle);
        }

        private void BuildTree(FileNode parentNode, string parentPath)
        {
            try
            {
                foreach (string dir in Directory.GetDirectories(parentPath))
                {
                    var dirNode = new FileNode(Path.GetFileName(dir), dir, true);
                    parentNode.Children.Add(dirNode);
                    BuildTree(dirNode, dir);
                }
                foreach (string file in Directory.GetFiles(parentPath))
                    parentNode.Children.Add(new FileNode(Path.GetFileName(file), file, false));
            }
            catch (Exception ex)
            {
                Log($"加载文件结构错误: {ex.Message}");
            }
        }

        private void FileTreeView_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            if (e.NewValue is FileNode selectedNode && !selectedNode.IsDirectory)
                OpenFileInTab(selectedNode.FullPath);
        }

        private void OpenFileInTab(string filePath)
        {
            if (openTabs.TryGetValue(filePath, out TabItem existingTab))
            {
                tabControl.SelectedItem = existingTab;
                return;
            }

            try
            {
                string fileContent = File.ReadAllText(filePath);
                var editor = new TextEditor
                {
                    FontFamily = new FontFamily("Consolas"),
                    FontSize = 14,
                    ShowLineNumbers = true,
                    WordWrap = true,
                    Document = new ICSharpCode.AvalonEdit.Document.TextDocument(fileContent)
                };

                string ext = Path.GetExtension(filePath).ToLower();
                if (ext == ".py") editor.SyntaxHighlighting = HighlightingManager.Instance.GetDefinition("Python");
                else if (ext == ".json") editor.SyntaxHighlighting = HighlightingManager.Instance.GetDefinition("JavaScript");

                var fileName = Path.GetFileName(filePath);
                var newTab = new TabItem { Tag = fileName, Content = editor, ToolTip = filePath };
                tabControl.Items.Add(newTab);
                tabControl.SelectedItem = newTab;
                openTabs[filePath] = newTab;

                originalCodeCache[filePath] = fileContent;
            }
            catch (Exception ex) { Log($"打开文件失败: {ex.Message}"); }
        }

        private void CloseTabButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is TabItem tab)
            {
                var filePath = openTabs.FirstOrDefault(x => x.Value == tab).Key;
                if (filePath != null)
                {
                    if (tab.Content is TextEditor editor)
                    {
                        string currentContent = editor.Document.Text;
                        if (originalCodeCache.TryGetValue(filePath, out string originalContent) &&
                            currentContent != originalContent)
                        {
                            var result = MessageBox.Show("文件已被修改，是否保存？", "保存更改",
                                MessageBoxButton.YesNoCancel, MessageBoxImage.Question);

                            if (result == MessageBoxResult.Yes)
                            {
                                File.WriteAllText(filePath, currentContent);
                                Log($"文件已保存: {Path.GetFileName(filePath)}");
                            }
                            else if (result == MessageBoxResult.Cancel)
                            {
                                return;
                            }
                        }
                    }

                    openTabs.Remove(filePath);
                    originalCodeCache.Remove(filePath);
                }
                tabControl.Items.Remove(tab);
            }
            e.Handled = true;
        }

        private void btnRun_Click(object sender, RoutedEventArgs e)
        {
            Log("开始运行模型...");
            Errors.Clear();
            Errors.Add(new ErrorItem
            {
                Line = 10,
                Column = 5,
                Message = "未找到模型文件",
                File = "model.py",
                Solution = "请添加模型文件或检查路径",
                IsResolved = false
            });
            ConsoleWriteLine("> run\n模型加载完成\n输入: 'Hello'\n输出: 'Hi there!'");
        }

        private void btnTest_Click(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("测试对话功能", "测试", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private void btnPack_Click(object sender, RoutedEventArgs e)
        {
            Log("开始打包SDK...");
            ConsoleWriteLine("> pack\n打包完成: olmo_sdk.zip");
        }

        private void Log(string message) => txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}\n");

        private void ConsoleWriteLine(string message) => txtConsole.AppendText($"{message}\n");

        private void ExecuteCommand()
        {
            string command = txtCommand.Text.Trim();
            if (string.IsNullOrEmpty(command)) return;

            ConsoleWriteLine($"> {command}");
            switch (command.ToLower())
            {
                case "help":
                    ConsoleWriteLine("可用命令: help, clear, run, test, train, pack, ai, struct");
                    break;
                case "clear": case "cls": txtConsole.Clear(); break;
                case "run": btnRun_Click(null, null); break;
                case "test": btnTest_Click(null, null); break;
                case "train": tabControlBottom.SelectedItem = trainingTab; break;
                case "pack": btnPack_Click(null, null); break;
                case "ai": tabControlBottom.SelectedItem = aiChatTab; break;
                case "struct": btnStructureCheck_Click(null, null); break;
                default: ConsoleWriteLine($"未知命令: {command}"); break;
            }
            txtCommand.Clear();
        }

        private void btnExecute_Click(object sender, RoutedEventArgs e) => ExecuteCommand();
        private void txtCommand_KeyDown(object sender, KeyEventArgs e) { if (e.Key == Key.Enter) ExecuteCommand(); }

        private void ExpandAllChildren(TreeViewItem item)
        {
            foreach (var childItem in item.Items)
                if (item.ItemContainerGenerator.ContainerFromItem(childItem) is TreeViewItem childContainer)
                {
                    childContainer.IsExpanded = true;
                    ExpandAllChildren(childContainer);
                }
        }

        private void NewFile_Click(object sender, RoutedEventArgs e)
        {
            if (fileTreeView.SelectedItem is FileNode selectedNode)
            {
                string parentPath = selectedNode.IsDirectory ? selectedNode.FullPath : Path.GetDirectoryName(selectedNode.FullPath);
                var dialog = new InputDialog("新建文件", "文件名:");
                if (dialog.ShowDialog() == true && !string.IsNullOrWhiteSpace(dialog.InputText))
                {
                    string fileName = dialog.InputText.Contains('.') ? dialog.InputText : $"{dialog.InputText}.py";
                    try
                    {
                        File.WriteAllText(Path.Combine(parentPath, fileName), "# 新文件");
                        RefreshFileTree();
                    }
                    catch (Exception ex) { Log($"创建文件失败: {ex.Message}"); }
                }
            }
        }

        private void NewFolder_Click(object sender, RoutedEventArgs e)
        {
            if (fileTreeView.SelectedItem is FileNode selectedNode)
            {
                string parentPath = selectedNode.IsDirectory ? selectedNode.FullPath : Path.GetDirectoryName(selectedNode.FullPath);
                var dialog = new InputDialog("新建文件夹", "文件夹名:");
                if (dialog.ShowDialog() == true && !string.IsNullOrWhiteSpace(dialog.InputText))
                {
                    try
                    {
                        Directory.CreateDirectory(Path.Combine(parentPath, dialog.InputText));
                        RefreshFileTree();
                    }
                    catch (Exception ex) { Log($"创建文件夹失败: {ex.Message}"); }
                }
            }
        }

        private void RenameItem_Click(object sender, RoutedEventArgs e)
        {
            if (fileTreeView.SelectedItem is FileNode selectedNode)
            {
                var dialog = new InputDialog("重命名", "新名称:", selectedNode.Name);
                if (dialog.ShowDialog() == true && !string.IsNullOrWhiteSpace(dialog.InputText))
                {
                    try
                    {
                        string newPath = Path.Combine(Path.GetDirectoryName(selectedNode.FullPath), dialog.InputText);
                        if (selectedNode.IsDirectory) Directory.Move(selectedNode.FullPath, newPath);
                        else File.Move(selectedNode.FullPath, newPath);
                        RefreshFileTree();
                    }
                    catch (Exception ex) { Log($"重命名失败: {ex.Message}"); }
                }
            }
        }

        private void DeleteItem_Click(object sender, RoutedEventArgs e)
        {
            if (fileTreeView.SelectedItem is FileNode selectedNode &&
                MessageBox.Show($"删除 '{selectedNode.Name}'?", "确认", MessageBoxButton.YesNo) == MessageBoxResult.Yes)
            {
                try
                {
                    if (selectedNode.IsDirectory) Directory.Delete(selectedNode.FullPath, true);
                    else File.Delete(selectedNode.FullPath);
                    RefreshFileTree();
                }
                catch (Exception ex) { Log($"删除失败: {ex.Message}"); }
            }
        }

        private void RefreshFileTree()
        {
            if (fileTreeView.Items.Count > 0 && fileTreeView.Items[0] is FileNode rootNode)
                LoadFileStructure(rootNode.FullPath);
        }

        private async void btnSend_Click(object sender, RoutedEventArgs e)
        {
            if (isProcessingAIRequest || string.IsNullOrWhiteSpace(txtChatInput.Text)) return;

            string message = txtChatInput.Text.Trim();
            AddChatMessage("你", message);
            txtChatInput.Clear();

            string filePath = "";
            if (fileTreeView.SelectedItem is FileNode selectedNode && !selectedNode.IsDirectory)
            {
                filePath = selectedNode.FullPath;
            }
            else if (tabControl.SelectedItem is TabItem currentTab && currentTab.Content is TextEditor)
            {
                filePath = currentTab.ToolTip?.ToString() ?? "";
            }

            if (string.IsNullOrEmpty(filePath))
            {
                AddChatMessage("AI助手", "请选择单个文件");
                return;
            }

            List<string> filesToProcess = new List<string> { filePath };
            AddChatMessage("AI助手", $"处理文件: {Path.GetFileName(filePath)}");

            isProcessingAIRequest = true;
            AddChatMessage("AI助手", "分析中...");

            try
            {
                string response = await SendToDeepSeek(message, filesToProcess);
                ProcessAIResponse(filesToProcess, response);
                AddChatMessage("AI助手", "分析完成! 查看修改建议");
            }
            catch (Exception ex)
            {
                AddChatMessage("AI助手", $"错误: {ex.Message}");
                Log($"AI请求失败: {ex.Message}");
            }
            finally { isProcessingAIRequest = false; }
        }

        private async Task<string> SendToDeepSeek(string userRequest, List<string> files)
        {
            using var client = new HttpClient();
            string apiKey = txtTeacherModelApiKey.Text;
            client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

            StringBuilder fileContents = new StringBuilder();
            foreach (var file in files)
            {
                try
                {
                    string fileContent = File.ReadAllText(file);
                    fileContents.AppendLine($"文件: {Path.GetFileName(file)}");
                    fileContents.AppendLine("```");
                    fileContents.AppendLine(fileContent);
                    fileContents.AppendLine("```\n");
                }
                catch (Exception ex) { Log($"读取文件失败: {file} - {ex.Message}"); }
            }

            var requestData = new
            {
                model = "deepseek-coder",
                messages = new[]
                {
                    new { role = "system", content = "你是一个专业软件开发助手。用户会提供代码文件或项目要求，返回修改后的完整代码。要求：1.只返回修改后的完整代码；2.保持原有结构；3.修复错误确保运行；4.优化保持功能；5.添加功能确保兼容；6.多个文件单独提供代码块，用// File: filename注释" },
                    new { role = "user", content = $"要求: {userRequest}\n\n代码文件:\n{fileContents}" }
                },
                temperature = 0.2,
                max_tokens = 8000
            };

            var requestContent = new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json");
            var response = await client.PostAsync(DeepSeekApiUrl, requestContent);

            if (!response.IsSuccessStatusCode)
            {
                string errorContent = await response.Content.ReadAsStringAsync();
                throw new Exception($"API错误: {response.StatusCode} - {errorContent}");
            }

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<DeepSeekResponse>(responseContent)?.choices?[0]?.message?.content ?? "";
        }

        private void ProcessAIResponse(List<string> originalFiles, string aiResponse)
        {
            var fileSections = Regex.Matches(aiResponse, @"\/\/ File:\s*([^\n]+?)\s*```[^\n]*\n([\s\S]*?)\n```", RegexOptions.IgnoreCase);
            if (fileSections.Count == 0)
            {
                AddChatMessage("AI助手", "未找到有效代码块");
                return;
            }

            foreach (Match match in fileSections)
            {
                string fileName = match.Groups[1].Value.Trim();
                string modifiedCode = match.Groups[2].Value;
                string filePath = originalFiles.FirstOrDefault(f => Path.GetFileName(f).Equals(fileName, StringComparison.OrdinalIgnoreCase))
                                ?? Path.Combine(Path.GetDirectoryName(originalFiles[0]), fileName);

                OpenCodeCompareView(filePath, modifiedCode, !File.Exists(filePath));
            }
        }

        private void OpenCodeCompareView(string filePath, string modifiedCode, bool isNewFile = false)
        {
            string originalCode = isNewFile ? "" : File.ReadAllText(filePath);
            var fileName = Path.GetFileName(filePath);
            var compareTab = new TabItem { Tag = $"对比: {fileName}", ToolTip = filePath };

            var grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition());
            grid.ColumnDefinitions.Add(new ColumnDefinition());
            grid.RowDefinitions.Add(new RowDefinition());
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            var originalEditor = CreateEditorWithContent(originalCode, filePath, true);
            var modifiedEditor = CreateEditorWithContent(modifiedCode, filePath);

            var buttonPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 10, 0, 5)
            };
            var btnAccept = new Button
            {
                Content = "接受修改",
                Width = 100,
                Margin = new Thickness(10),
                Tag = new Tuple<string, string, bool>(filePath, modifiedCode, isNewFile)
            };
            btnAccept.Click += BtnAcceptChanges_Click;

            var btnReject = new Button
            {
                Content = "拒绝修改",
                Width = 100,
                Margin = new Thickness(10),
                Tag = filePath
            };
            btnReject.Click += BtnRejectChanges_Click;

            buttonPanel.Children.Add(btnAccept);
            buttonPanel.Children.Add(btnReject);

            grid.Children.Add(originalEditor);
            grid.Children.Add(modifiedEditor);
            grid.Children.Add(buttonPanel);
            Grid.SetColumn(modifiedEditor, 1);
            Grid.SetRow(buttonPanel, 1);
            Grid.SetColumnSpan(buttonPanel, 2);

            compareTab.Content = grid;
            tabControl.Items.Add(compareTab);
            tabControl.SelectedItem = compareTab;
        }

        private TextEditor CreateEditorWithContent(string content, string filePath, bool isReadOnly = false)
        {
            var editor = new TextEditor
            {
                Document = new ICSharpCode.AvalonEdit.Document.TextDocument(content),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 14,
                ShowLineNumbers = true,
                WordWrap = false,
                IsReadOnly = isReadOnly,
                Background = isReadOnly ? Brushes.LavenderBlush : Brushes.Honeydew
            };

            string ext = Path.GetExtension(filePath).ToLower();
            if (ext == ".py") editor.SyntaxHighlighting = HighlightingManager.Instance.GetDefinition("Python");
            else if (ext == ".json") editor.SyntaxHighlighting = HighlightingManager.Instance.GetDefinition("JavaScript");

            return editor;
        }

        private void BtnAcceptChanges_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is Tuple<string, string, bool> data)
            {
                string filePath = data.Item1;
                string modifiedCode = data.Item2;
                bool isNewFile = data.Item3;

                try
                {
                    if (isNewFile) Directory.CreateDirectory(Path.GetDirectoryName(filePath));
                    File.WriteAllText(filePath, modifiedCode);

                    if (isNewFile)
                    {
                        RefreshFileTree();
                    }
                    else if (openTabs.TryGetValue(filePath, out TabItem tab) && tab.Content is TextEditor editor)
                    {
                        editor.Document.Text = modifiedCode;
                        originalCodeCache[filePath] = modifiedCode;
                    }

                    AddChatMessage("系统", $"已保存: {Path.GetFileName(filePath)}");
                }
                catch (Exception ex) { AddChatMessage("系统", $"保存失败: {ex.Message}"); }
                CloseCompareTab(filePath);
            }
        }

        private void BtnRejectChanges_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string filePath)
                CloseCompareTab(filePath);
        }

        private void CloseCompareTab(string filePath)
        {
            var compareTab = tabControl.Items.OfType<TabItem>()
                .FirstOrDefault(t => t.Tag?.ToString()?.StartsWith("对比:") == true && t.ToolTip?.ToString() == filePath);
            if (compareTab != null) tabControl.Items.Remove(compareTab);
        }

        private void txtChatInput_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter && (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control)
            {
                btnSend_Click(sender, e);
                e.Handled = true;
            }
        }

        private void AddChatMessage(string sender, string message)
        {
            lstChat.Items.Add($"[{DateTime.Now:HH:mm:ss}] {sender}: {message}");
            lstChat.ScrollIntoView(lstChat.Items[lstChat.Items.Count - 1]);
        }

        private async void BtnAIFix_Click(object sender, RoutedEventArgs e)
        {
            if (sender is TextBlock textBlock && textBlock.Tag is ErrorItem error)
            {
                try
                {
                    // 标记错误为正在解决
                    error.IsResolving = true;
                    RefreshErrorList();

                    if (error.Message.Contains("未找到模型文件"))
                    {
                        string modelFilePath = Path.Combine(currentProjectRoot, "model.py");

                        if (File.Exists(modelFilePath))
                        {
                            Log($"模型文件已存在: {modelFilePath}");
                            MessageBox.Show("模型文件已存在，请检查路径配置", "错误", MessageBoxButton.OK, MessageBoxImage.Warning);
                            return;
                        }

                        string modelCode = @"import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 示例使用
if __name__ == '__main__':
    model = SimpleModel(128, 256, 10)
    print('模型结构:')
    print(model)
    print('\n模型参数数量:', sum(p.numel() for p in model.parameters()))";

                        try
                        {
                            File.WriteAllText(modelFilePath, modelCode);
                            Log($"已创建模型文件: {modelFilePath}");
                            RefreshFileTree();
                            OpenFileInTab(modelFilePath);
                            error.IsResolved = true;
                            Errors.Remove(error);
                            MessageBox.Show("模型文件已创建并添加到项目中", "成功", MessageBoxButton.OK, MessageBoxImage.Information);
                        }
                        catch (Exception ex)
                        {
                            Log($"创建模型文件失败: {ex.Message}");
                            MessageBox.Show($"创建模型文件失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                        }
                    }
                    else
                    {
                        string fileContent = File.ReadAllText(error.File);

                        using var client = new HttpClient();
                        string apiKey = txtTeacherModelApiKey.Text;
                        client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

                        var requestData = new
                        {
                            model = "deepseek-coder",
                            messages = new[]
                            {
                                new { role = "system", content = "你是一个代码修复助手。用户会提供代码文件和错误信息，请修复错误并返回整个文件修复后的内容。要求：1.只返回修改后的完整代码；2.保持原有结构；3.修复错误确保运行。" },
                                new { role = "user", content = $"错误信息：在文件{error.File}的第{error.Line}行，第{error.Column}列：{error.Message}\n\n文件内容：\n```\n{fileContent}\n```" }
                            },
                            temperature = 0.1,
                            max_tokens = 4000
                        };

                        var requestContent = new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json");
                        var response = await client.PostAsync(DeepSeekApiUrl, requestContent);

                        if (!response.IsSuccessStatusCode)
                        {
                            Log($"修复错误时API错误: {response.StatusCode}");
                            return;
                        }

                        string responseContent = await response.Content.ReadAsStringAsync();
                        string aiResponse = JsonConvert.DeserializeObject<DeepSeekResponse>(responseContent)?.choices?[0]?.message?.content ?? "";

                        var codeMatch = Regex.Match(aiResponse, @"```[^\n]*\n([\s\S]*?)\n```");
                        if (codeMatch.Success)
                        {
                            string fixedCode = codeMatch.Groups[1].Value;
                            OpenCodeCompareView(error.File, fixedCode);
                            Log($"已生成修复方案: {error.File}");
                            error.IsResolved = true;
                        }
                        else
                        {
                            Log("未找到修复后的代码");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"修复错误失败: {ex.Message}");
                }
                finally
                {
                    error.IsResolving = false;
                    RefreshErrorList();
                }
            }
        }

        private async void btnStartTraining_Click(object sender, RoutedEventArgs e)
        {
            if (isTraining) return;

            isTraining = true;
            trainingCancellationTokenSource = new CancellationTokenSource();
            btnStartTraining.IsEnabled = false;
            btnStopTraining.IsEnabled = true;

            AppendTrainingLog("开始训练...");
            AppendTrainingLog("训练中...");

            try
            {
                for (int epoch = 1; epoch <= 10; epoch++)
                {
                    if (trainingCancellationTokenSource.Token.IsCancellationRequested) break;
                    AppendTrainingLog($"Epoch {epoch}/10 - 训练中...");
                    await Task.Delay(1500, trainingCancellationTokenSource.Token);
                    AppendTrainingLog($"Epoch {epoch} 结果 - 损失: {2.5 - (epoch * 0.2):F4}, 准确率: {epoch * 0.08:P}");
                }
                AppendTrainingLog(trainingCancellationTokenSource.Token.IsCancellationRequested ? "训练已取消" : "训练完成!");
            }
            catch (OperationCanceledException) { AppendTrainingLog("训练已取消"); }
            catch (Exception ex) { AppendTrainingLog($"训练错误: {ex.Message}"); }
            finally
            {
                isTraining = false;
                btnStartTraining.IsEnabled = true;
                btnStopTraining.IsEnabled = false;
            }
        }

        private void btnStopTraining_Click(object sender, RoutedEventArgs e)
        {
            trainingCancellationTokenSource?.Cancel();
            AppendTrainingLog("正在停止训练...");
        }

        private void AppendTrainingLog(string message)
        {
            Dispatcher.Invoke(() =>
            {
                txtTrainingLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}\n");
                txtTrainingLog.ScrollToEnd();
            });
        }

        private async void btnGenerateData_Click(object sender, RoutedEventArgs e)
        {
            AppendTrainingLog("生成训练数据中...");
            try
            {
                using var client = new HttpClient();
                string apiKey = txtTeacherModelApiKey.Text;
                client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

                var requestData = new
                {
                    model = "deepseek-coder",
                    messages = new[]
                    {
                        new { role = "user", content = $"生成10条高质量对话数据(JSON格式)" }
                    },
                    temperature = 0.3,
                    max_tokens = 2000
                };

                var requestContent = new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json");
                var response = await client.PostAsync(DeepSeekApiUrl, requestContent);

                if (!response.IsSuccessStatusCode)
                    throw new Exception($"API错误: {response.StatusCode}");

                string responseContent = await response.Content.ReadAsStringAsync();
                string aiResponse = JsonConvert.DeserializeObject<DeepSeekResponse>(responseContent)?.choices?[0]?.message?.content ?? "";

                var jsonMatch = Regex.Match(aiResponse, @"```json\n([\s\S]*?)\n```");
                AppendTrainingLog(jsonMatch.Success ? jsonMatch.Groups[1].Value : aiResponse);
            }
            catch (Exception ex) { AppendTrainingLog($"生成失败: {ex.Message}"); }
        }

        private void btnSaveTrainingConfig_Click(object sender, RoutedEventArgs e)
        {
            string saveDir = currentProjectRoot;
            if (fileTreeView.SelectedItem is FileNode selectedNode)
            {
                saveDir = selectedNode.IsDirectory ? selectedNode.FullPath : Path.GetDirectoryName(selectedNode.FullPath);
            }

            if (string.IsNullOrEmpty(saveDir))
            {
                AppendTrainingLog("请先选择一个文件夹");
                return;
            }

            string configPath = Path.Combine(saveDir, "training_config.json");
            string trainScriptPath = Path.Combine(saveDir, "train.py");

            try
            {
                File.WriteAllText(configPath, JsonConvert.SerializeObject(new
                {
                    Timestamp = DateTime.Now
                }, Formatting.Indented));

                string trainScript = @"# 自动生成的训练脚本
import torch
from torch.utils.data import DataLoader
from src.model.main_model import create_model
import yaml
import os
import time

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    # 加载配置
    config = load_config('configs/base_config.yaml')
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout']
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(training_config['epochs']):
        start_time = time.time()
        total_loss = 0
        
        # 模拟训练
        for i in range(100):
            optimizer.zero_grad()
            # 假设输入和输出
            inputs = torch.randint(0, model_config['vocab_size'], (training_config['batch_size'], model_config['max_seq_len']), device=device)
            targets = torch.randint(0, model_config['vocab_size'], (training_config['batch_size'], model_config['max_seq_len']), device=device)
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model_config['vocab_size']), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / 100
        elapsed = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{training_config[""epochs""]} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s')
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, os.path.join(training_config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pt'))

if __name__ == '__main__':
    train()
";
                File.WriteAllText(trainScriptPath, trainScript);

                AppendTrainingLog($"配置已保存: {configPath}");
                AppendTrainingLog($"训练脚本已生成: {trainScriptPath}");

                RefreshFileTree();
                OpenFileInTab(trainScriptPath);
            }
            catch (Exception ex) { AppendTrainingLog($"保存失败: {ex.Message}"); }
        }

        private void btnApplyWeights_Click(object sender, RoutedEventArgs e)
        {
            Log($"应用权重策略: {((ComboBoxItem)cmbInitStrategy.SelectedItem).Content}");
            if (chkKnowledgeDistillation.IsChecked == true)
                Log($"使用知识蒸馏, API密钥: {txtTeacherModelApiKey.Text}");
            UpdateWeightPreview();
        }

        private void UpdateWeightPreview()
        {
            dgWeights.ItemsSource = new List<WeightLayer>
            {
                new WeightLayer() { Layer = "embedding", Type = "Embedding", Initialization = "Normal(0, 0.02)" },
                new WeightLayer() { Layer = "transformer.0", Type = "Linear", Initialization = "Xavier Uniform" },
                new WeightLayer() { Layer = "transformer.1", Type = "LayerNorm", Initialization = "Normal(1, 0.02)" },
                new WeightLayer() { Layer = "transformer.2", Type = "Linear", Initialization = "Kaiming Normal" },
                new WeightLayer() { Layer = "lm_head", Type = "Linear", Initialization = "Zero" }
            };
        }

        private void btnExportWeights_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new SaveFileDialog { Filter = "权重文件|*.weights|JSON文件|*.json" };
            if (dialog.ShowDialog() == true)
            {
                try
                {
                    File.WriteAllText(dialog.FileName, JsonConvert.SerializeObject(new
                    {
                        Strategy = ((ComboBoxItem)cmbInitStrategy.SelectedItem).Content,
                        KnowledgeDistillation = chkKnowledgeDistillation.IsChecked,
                        TeacherModelApiKey = txtTeacherModelApiKey.Text,
                        CustomWeights = txtLayerWeights.Text
                    }, Formatting.Indented));
                    Log($"权重配置已导出: {dialog.FileName}");
                }
                catch (Exception ex) { Log($"导出失败: {ex.Message}"); }
            }
        }

        private async void btnOptimizeWeights_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(txtLayerWeights.Text)) return;

            try
            {
                using var client = new HttpClient();
                string apiKey = txtTeacherModelApiKey.Text;
                client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

                var requestData = new
                {
                    model = "deepseek-coder",
                    messages = new[]
                    {
                        new { role = "user", content = "优化以下权重配置:\n" + txtLayerWeights.Text + "\n要求: 1.改进初始化 2.提高稳定性 3.保持容量" }
                    },
                    temperature = 0.1,
                    max_tokens = 1000
                };

                var requestContent = new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json");
                var response = await client.PostAsync(DeepSeekApiUrl, requestContent);

                if (!response.IsSuccessStatusCode)
                    throw new Exception($"API错误: {response.StatusCode}");

                string responseContent = await response.Content.ReadAsStringAsync();
                string aiResponse = JsonConvert.DeserializeObject<DeepSeekResponse>(responseContent)?.choices?[0]?.message?.content ?? "";

                var jsonMatch = Regex.Match(aiResponse, @"```json\n([\s\S]*?)\n```");
                if (jsonMatch.Success)
                {
                    string optimizedWeights = jsonMatch.Groups[1].Value;
                    OpenCodeCompareView("权重配置", txtLayerWeights.Text, optimizedWeights);
                    Log("权重已优化，请查看对比");
                }
                else
                {
                    txtLayerWeights.Text = aiResponse;
                    Log("未找到JSON格式权重配置");
                }
            }
            catch (Exception ex) { Log($"优化失败: {ex.Message}"); }
        }

        private void OpenCodeCompareView(string fileName, string originalContent, string modifiedContent)
        {
            var compareTab = new TabItem { Tag = $"对比: {fileName}" };

            var grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition());
            grid.ColumnDefinitions.Add(new ColumnDefinition());
            grid.RowDefinitions.Add(new RowDefinition());
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            var originalEditor = new TextEditor
            {
                Document = new ICSharpCode.AvalonEdit.Document.TextDocument(originalContent),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 14,
                ShowLineNumbers = true,
                IsReadOnly = true,
                Background = Brushes.LavenderBlush
            };

            var modifiedEditor = new TextEditor
            {
                Document = new ICSharpCode.AvalonEdit.Document.TextDocument(modifiedContent),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 14,
                ShowLineNumbers = true,
                Background = Brushes.Honeydew
            };

            var buttonPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 10, 0, 5)
            };
            var btnAccept = new Button
            {
                Content = "接受优化",
                Width = 100,
                Margin = new Thickness(10),
                Tag = modifiedContent
            };
            btnAccept.Click += (s, ev) => {
                txtLayerWeights.Text = modifiedContent;
                Log("已接受权重优化");
                CloseCompareTab(compareTab);
            };

            var btnReject = new Button
            {
                Content = "拒绝优化",
                Width = 100,
                Margin = new Thickness(10),
            };
            btnReject.Click += (s, ev) => CloseCompareTab(compareTab);

            buttonPanel.Children.Add(btnAccept);
            buttonPanel.Children.Add(btnReject);

            grid.Children.Add(originalEditor);
            grid.Children.Add(modifiedEditor);
            grid.Children.Add(buttonPanel);
            Grid.SetColumn(modifiedEditor, 1);
            Grid.SetRow(buttonPanel, 1);
            Grid.SetColumnSpan(buttonPanel, 2);

            compareTab.Content = grid;
            tabControl.Items.Add(compareTab);
            tabControl.SelectedItem = compareTab;
        }

        private void CloseCompareTab(TabItem tab)
        {
            tabControl.Items.Remove(tab);
        }

        private async void btnGenerateWeights_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(txtWeightIdea.Text))
            {
                MessageBox.Show("请描述您的权重想法");
                return;
            }

            try
            {
                using var client = new HttpClient();
                string apiKey = txtTeacherModelApiKey.Text;
                client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

                var requestData = new
                {
                    model = "deepseek-coder",
                    messages = new[]
                    {
                        new {
                            role = "system",
                            content = "你是一个深度学习专家。用户会描述权重初始化策略的想法，你需要生成对应的JSON格式权重配置。"
                        },
                        new {
                            role = "user",
                            content = $"权重想法描述: {txtWeightIdea.Text}\n\n请生成JSON格式的权重配置:"
                        }
                    },
                    temperature = 0.1,
                    max_tokens = 1000
                };

                var requestContent = new StringContent(JsonConvert.SerializeObject(requestData), Encoding.UTF8, "application/json");
                var response = await client.PostAsync(DeepSeekApiUrl, requestContent);

                if (!response.IsSuccessStatusCode)
                    throw new Exception($"API错误: {response.StatusCode}");

                string responseContent = await response.Content.ReadAsStringAsync();
                string aiResponse = JsonConvert.DeserializeObject<DeepSeekResponse>(responseContent)?.choices?[0]?.message?.content ?? "";

                var jsonMatch = Regex.Match(aiResponse, @"```json\n([\s\S]*?)\n```");
                if (jsonMatch.Success)
                {
                    string weightsJson = jsonMatch.Groups[1].Value;
                    txtLayerWeights.Text = weightsJson;
                    CreateWeightsFile(weightsJson);
                    Log("权重已生成并保存");
                }
                else
                {
                    txtLayerWeights.Text = aiResponse;
                    Log("未找到JSON格式权重配置");
                }
            }
            catch (Exception ex)
            {
                Log($"权重生成失败: {ex.Message}");
                txtLayerWeights.Text = $"错误: {ex.Message}";
            }
        }

        private void CreateWeightsFile(string weightsJson)
        {
            string saveDir = currentProjectRoot;
            if (fileTreeView.SelectedItem is FileNode selectedNode)
            {
                saveDir = selectedNode.IsDirectory ? selectedNode.FullPath : Path.GetDirectoryName(selectedNode.FullPath);
            }

            if (string.IsNullOrEmpty(saveDir))
            {
                MessageBox.Show("请先选择一个文件夹");
                return;
            }

            string fileName = "weights_config.json";
            string filePath = Path.Combine(saveDir, fileName);

            try
            {
                File.WriteAllText(filePath, weightsJson);
                Log($"权重文件已创建: {filePath}");
                RefreshFileTree();
                OpenFileInTab(filePath);
            }
            catch (Exception ex)
            {
                Log($"创建权重文件失败: {ex.Message}");
            }
        }

        private void btnStructureTree_Click(object sender, RoutedEventArgs e)
        {
            structureTreePanel.Visibility = Visibility.Visible;
            importPanel.Visibility = Visibility.Collapsed;
        }

        private void btnCancelStructure_Click(object sender, RoutedEventArgs e)
        {
            structureTreePanel.Visibility = Visibility.Collapsed;
        }

        private void btnGenerateStructure_Click(object sender, RoutedEventArgs e)
        {
            string requirement = txtStructureRequirement.Text.Trim();
            if (string.IsNullOrWhiteSpace(requirement))
            {
                MessageBox.Show("请输入模型结构要求");
                return;
            }

            var dialog = new WinForms.FolderBrowserDialog();
            if (dialog.ShowDialog() == WinForms.DialogResult.OK)
            {
                string savePath = dialog.SelectedPath;
                try
                {
                    StructureGenerator generator = new StructureGenerator();
                    generator.GenerateStructure(requirement, savePath);

                    LoadFileStructure(savePath);
                    currentProjectRoot = savePath;

                    Log($"已生成模型结构树: {savePath}");
                    structureTreePanel.Visibility = Visibility.Collapsed;
                }
                catch (Exception ex)
                {
                    Log($"生成结构树失败: {ex.Message}");
                    MessageBox.Show($"生成结构树失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        // 修复结构树检测中的空值转换问题
        // MainWindow.xaml.cs (部分修改)
        private async void btnStructureCheck_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentProjectRoot))
            {
                MessageBox.Show("请先导入项目或生成结构树");
                return;
            }

            try
            {
                Log("开始结构树检测与修复...");
                AddChatMessage("系统", "开始检测并修复项目结构...");

                // 接收元组返回值 (操作计数, 结果字符串)
                (int operationsApplied, string result) = await structureValidator.ValidateAndFixStructure(currentProjectRoot);

                Log(result);
                AddChatMessage("AI助手", result);

                // 刷新文件树
                LoadFileStructure(currentProjectRoot);

                MessageBox.Show(result, "结构树修复完成", MessageBoxButton.OK,
                                operationsApplied > 0 ? MessageBoxImage.Information : MessageBoxImage.Warning);
            }
            catch (Exception ex)
            {
                string errorMsg = $"结构树检测失败: {ex.Message}";
                Log(errorMsg);
                AddChatMessage("AI助手", errorMsg);
                MessageBox.Show(errorMsg, "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void RefreshErrorList()
        {
            // 创建临时列表以触发UI更新
            var tempList = new ObservableCollection<ErrorItem>(Errors);
            Errors.Clear();
            foreach (var item in tempList)
            {
                Errors.Add(item);
            }
        }
    }

    public class FileNode : INotifyPropertyChanged
    {
        public string Name { get; set; }
        public string FullPath { get; set; }
        public bool IsDirectory { get; set; }
        public ObservableCollection<FileNode> Children { get; set; } = new ObservableCollection<FileNode>();
        public string DisplayName => Name;

        public FileNode(string name, string fullPath, bool isDirectory)
        {
            Name = name;
            FullPath = fullPath;
            IsDirectory = isDirectory;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    public class ErrorItem : INotifyPropertyChanged
    {
        private bool _isResolved;
        private bool _isResolving;

        public int Line { get; set; }
        public int Column { get; set; }
        public string Message { get; set; }
        public string File { get; set; }
        public string Solution { get; set; }

        public bool IsResolved
        {
            get => _isResolved;
            set
            {
                if (_isResolved != value)
                {
                    _isResolved = value;
                    OnPropertyChanged(nameof(IsResolved));
                }
            }
        }

        public bool IsResolving
        {
            get => _isResolving;
            set
            {
                if (_isResolving != value)
                {
                    _isResolving = value;
                    OnPropertyChanged(nameof(IsResolving));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    public class DeepSeekResponse
    {
        public List<Choice> choices { get; set; }
    }

    public class Choice
    {
        public Message message { get; set; }
    }

    public class Message
    {
        public string content { get; set; }
    }

    public class WeightLayer
    {
        public string Layer { get; set; }
        public string Type { get; set; }
        public string Initialization { get; set; }
    }

    public class InputDialog : Window
    {
        public string InputText { get; private set; }

        public InputDialog(string title, string prompt, string defaultText = "")
        {
            Title = title;
            Width = 300;
            Height = 180;
            WindowStartupLocation = WindowStartupLocation.CenterOwner;

            var stack = new StackPanel { Margin = new Thickness(10) };
            stack.Children.Add(new TextBlock { Text = prompt });

            var textBox = new TextBox { Text = defaultText, Margin = new Thickness(0, 5, 0, 10) };
            stack.Children.Add(textBox);

            var buttonPanel = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
            var okButton = new Button { Content = "确定", Width = 80, Margin = new Thickness(5) };
            okButton.Click += (s, e) => { InputText = textBox.Text; DialogResult = true; };

            var cancelButton = new Button { Content = "取消", Width = 80, Margin = new Thickness(5) };
            cancelButton.Click += (s, e) => DialogResult = false;

            buttonPanel.Children.Add(okButton);
            buttonPanel.Children.Add(cancelButton);
            stack.Children.Add(buttonPanel);

            Content = stack;
        }
    }
}