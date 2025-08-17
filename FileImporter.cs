using Microsoft.Win32;
using System;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using WinForms = System.Windows.Forms;
using SharpCompress.Archives;

namespace AI生成AI
{
    public static class FileImporter
    {
        public static void BrowseZip(TextBox txtZipPath)
        {
            try
            {
                var dialog = new OpenFileDialog { Filter = "ZIP文件|*.zip" };
                if (dialog.ShowDialog() == true) txtZipPath.Text = dialog.FileName;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"浏览ZIP文件时出错: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public static void BrowseFolder(TextBox txtFolderPath)
        {
            try
            {
                var dialog = new WinForms.FolderBrowserDialog();
                if (dialog.ShowDialog() == WinForms.DialogResult.OK) txtFolderPath.Text = dialog.SelectedPath;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"浏览文件夹时出错: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public static void ExtractZip(string zipPath, string extractPath, TextBox txtLog)
        {
            if (string.IsNullOrWhiteSpace(zipPath) || string.IsNullOrWhiteSpace(extractPath))
            {
                MessageBox.Show("请选择ZIP文件和解压目录");
                return;
            }

            try
            {
                using (var archive = ArchiveFactory.Open(zipPath))
                {
                    int fileCount = 0;
                    foreach (var entry in archive.Entries.Where(entry => !entry.IsDirectory))
                    {
                        try
                        {
                            string targetPath = Path.Combine(extractPath, entry.Key);
                            Directory.CreateDirectory(Path.GetDirectoryName(targetPath));
                            using (var stream = entry.OpenEntryStream())
                            using (var fileStream = File.Create(targetPath))
                                stream.CopyTo(fileStream);
                            fileCount++;
                        }
                        catch (Exception ex)
                        {
                            txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 解压文件 {entry.Key} 失败: {ex.Message}\n");
                        }
                    }
                    txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 解压完成! 共解压 {fileCount} 个文件\n");
                }
            }
            catch (Exception ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 解压失败: {ex.Message}\n");
                MessageBox.Show($"解压ZIP文件时出错: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public static void BrowseFolderImport(TextBox txtFolderPath)
        {
            try
            {
                var dialog = new WinForms.FolderBrowserDialog();
                if (dialog.ShowDialog() == WinForms.DialogResult.OK) txtFolderPath.Text = dialog.SelectedPath;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"浏览文件夹时出错: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public static void ImportFolder(string folderPath, TreeView fileTreeView, TextBox txtLog, ref string currentProjectRoot)
        {
            if (string.IsNullOrWhiteSpace(folderPath))
            {
                MessageBox.Show("请选择要导入的文件夹");
                return;
            }

            try
            {
                LoadFileStructure(folderPath, fileTreeView, txtLog);
                currentProjectRoot = folderPath;
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 项目文件夹已导入: {folderPath}\n");
            }
            catch (Exception ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 导入文件夹失败: {ex.Message}\n");
                MessageBox.Show($"导入文件夹时出错: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public static void BrowseFileImport(TextBox txtFilePath)
        {
            try
            {
                var dialog = new OpenFileDialog();
                if (dialog.ShowDialog() == true) txtFilePath.Text = dialog.FileName;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"浏览文件时出错: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public static void ImportFile(string filePath, TreeView fileTreeView, TextBox txtLog, ref string currentProjectRoot)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                MessageBox.Show("请选择要导入的文件");
                return;
            }

            try
            {
                // 如果还没有项目根目录，先让用户选择
                if (string.IsNullOrEmpty(currentProjectRoot))
                {
                    var folderDialog = new WinForms.FolderBrowserDialog();
                    if (folderDialog.ShowDialog() == WinForms.DialogResult.OK)
                    {
                        currentProjectRoot = folderDialog.SelectedPath;
                    }
                    else
                    {
                        return;
                    }
                }

                string fileName = Path.GetFileName(filePath);
                string destPath = Path.Combine(currentProjectRoot, fileName);

                // 确保目标目录存在
                Directory.CreateDirectory(Path.GetDirectoryName(destPath));

                // 复制文件（如果已存在则覆盖）
                File.Copy(filePath, destPath, true);

                // 刷新文件树
                LoadFileStructure(currentProjectRoot, fileTreeView, txtLog);
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 文件已导入: {fileName}\n");
            }
            catch (Exception ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 导入文件失败: {ex.Message}\n");
                MessageBox.Show($"导入文件时出错: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private static void LoadFileStructure(string rootPath, TreeView fileTreeView, TextBox txtLog)
        {
            fileTreeView.Items.Clear();

            if (!Directory.Exists(rootPath))
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 目录不存在: {rootPath}\n");
                return;
            }

            try
            {
                var rootNode = new FileNode(Path.GetFileName(rootPath), rootPath, true);
                BuildTree(rootNode, rootPath, txtLog);
                fileTreeView.Items.Add(rootNode);
            }
            catch (Exception ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 加载文件结构失败: {ex.Message}\n");
            }
        }

        private static void BuildTree(FileNode parentNode, string parentPath, TextBox txtLog)
        {
            try
            {
                // 获取所有子目录
                foreach (string dir in Directory.GetDirectories(parentPath))
                {
                    try
                    {
                        var dirNode = new FileNode(Path.GetFileName(dir), dir, true);
                        parentNode.Children.Add(dirNode);
                        BuildTree(dirNode, dir, txtLog);
                    }
                    catch (UnauthorizedAccessException ex)
                    {
                        txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 访问目录被拒绝: {dir} - {ex.Message}\n");
                    }
                    catch (Exception ex)
                    {
                        txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 处理目录时出错: {dir} - {ex.Message}\n");
                    }
                }

                // 获取所有文件
                foreach (string file in Directory.GetFiles(parentPath))
                {
                    try
                    {
                        parentNode.Children.Add(new FileNode(Path.GetFileName(file), file, false));
                    }
                    catch (Exception ex)
                    {
                        txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 处理文件时出错: {file} - {ex.Message}\n");
                    }
                }
            }
            catch (DirectoryNotFoundException ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 目录不存在: {parentPath} - {ex.Message}\n");
            }
            catch (UnauthorizedAccessException ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 访问目录被拒绝: {parentPath} - {ex.Message}\n");
            }
            catch (PathTooLongException ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 路径过长: {parentPath} - {ex.Message}\n");
            }
            catch (IOException ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] IO错误: {parentPath} - {ex.Message}\n");
            }
            catch (Exception ex)
            {
                txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] 加载文件结构时发生未知错误: {parentPath} - {ex.Message}\n");
            }
        }
    }
}