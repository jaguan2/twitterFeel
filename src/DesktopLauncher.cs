// Native entry point: opens the installed desktop app, without a server/browser.
using System;
using System.Diagnostics;
using System.IO;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class Program
{
    private static readonly object LogGate = new object();
    private static string FindRoot()
    {
        var directory = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
        while (directory != null)
        {
            if (File.Exists(Path.Combine(directory.FullName, @"src\desktop_app.py"))) return directory.FullName;
            directory = directory.Parent;
        }
        throw new InvalidOperationException("Keep TwitterFeel.exe inside its project folder. The src/desktop_app.py file could not be found.");
    }
    private static void Log(string path, string line)
    {
        if (line == null) return;
        lock (LogGate) { try { File.AppendAllText(path, DateTime.UtcNow.ToString("o") + " " + line + Environment.NewLine); } catch (IOException) { } }
    }
    [STAThread]
    public static int Main(string[] args)
    {
        bool smoke = args.Length == 2 && args[0] == "--smoke-test";
        string reportPath = smoke ? Path.GetFullPath(args[1]) : null;
        try
        {
            if (args.Length != 0 && !smoke) throw new InvalidOperationException("Usage: TwitterFeel.exe [--smoke-test report.json]");
            string root = FindRoot();
            string python = Path.Combine(root, @".venv-workbench\Scripts\pythonw.exe");
            if (!File.Exists(python)) throw new InvalidOperationException("The desktop environment is missing. Follow the desktop setup commands in README.md, then reopen TwitterFeel.exe.");
            string logs = Path.Combine(root, @"runs\workbench\logs");
            Directory.CreateDirectory(logs);
            string log = Path.Combine(logs, "launcher-" + DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffffffZ") + ".log");
            string command = "\"" + Path.Combine(root, @"src\desktop_app.py") + "\"";
            if (smoke) command += " --smoke-test \"" + reportPath + "\"";
            var info = new ProcessStartInfo(python, command);
            info.WorkingDirectory = root;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            info.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            using (var child = new Process())
            {
                child.StartInfo = info;
                child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { Log(log, e.Data); };
                child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { Log(log, e.Data); };
                child.Start();
                child.BeginOutputReadLine();
                child.BeginErrorReadLine();
                child.WaitForExit();
                int result = child.ExitCode;
                if (result != 0 && !smoke) MessageBox.Show("TwitterFeel could not complete startup. See log:\n" + log, "TwitterFeel", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return result;
            }
        }
        catch (Exception e)
        {
            if (smoke)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(reportPath));
                File.WriteAllText(reportPath, new JavaScriptSerializer().Serialize(new { success = false, mode = "native_webview2", error = e.Message }));
            }
            else MessageBox.Show(e.Message, "TwitterFeel", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
