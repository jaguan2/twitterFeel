@echo off
setlocal
cd /d "%~dp0"
set "CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%CSC%" set "CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe"
if not exist "%CSC%" (
  echo .NET Framework compiler not found. Install .NET Framework 4.8 developer tools.
  exit /b 1
)
".venv-workbench\Scripts\python.exe" src\build_version.py
if errorlevel 1 exit /b 1
"%CSC%" /nologo /target:winexe /optimize+ /out:TwitterFeel.exe /win32icon:src\desktop_ui\app.ico /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Web.Extensions.dll src\DesktopLauncher.cs runs\workbench\build-version.cs
if errorlevel 1 exit /b 1
echo Built %CD%\TwitterFeel.exe
