# FunASR转录服务器开机自启设置脚本
# 需要以管理员权限运行

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Status
)

$ServiceName = "FunASRTranscriber"
$ServiceDisplayName = "FunASR转录服务器"
$ScriptPath = Split-Path -Parent $PSScriptRoot
$BackgroundBat = Join-Path $PSScriptRoot "start_server_background.bat"

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-AutoStart {
    Write-Host "正在设置开机自启..." -ForegroundColor Green
    
    # 方法1: 注册表启动项
    $RegPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
    try {
        Set-ItemProperty -Path $RegPath -Name $ServiceName -Value "`"$BackgroundBat`"" -Force
        Write-Host "✓ 已添加到注册表启动项" -ForegroundColor Green
        
        # 方法2: 任务计划程序（更可靠）
        $Action = New-ScheduledTaskAction -Execute $BackgroundBat
        $Trigger = New-ScheduledTaskTrigger -AtStartup
        $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        $Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
        
        Register-ScheduledTask -TaskName $ServiceName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description $ServiceDisplayName -Force
        Write-Host "✓ 已添加到任务计划程序" -ForegroundColor Green
        
        Write-Host "开机自启设置完成！" -ForegroundColor Yellow
        Write-Host "服务将在下次重启时自动启动" -ForegroundColor Yellow
        
    } catch {
        Write-Host "❌ 设置失败: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Uninstall-AutoStart {
    Write-Host "正在移除开机自启..." -ForegroundColor Yellow
    
    # 移除注册表启动项
    $RegPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
    try {
        Remove-ItemProperty -Path $RegPath -Name $ServiceName -ErrorAction SilentlyContinue
        Write-Host "✓ 已从注册表启动项移除" -ForegroundColor Green
    } catch {
        Write-Host "! 注册表项不存在或已移除" -ForegroundColor Yellow
    }
    
    # 移除任务计划
    try {
        Unregister-ScheduledTask -TaskName $ServiceName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "✓ 已从任务计划程序移除" -ForegroundColor Green
    } catch {
        Write-Host "! 任务计划不存在或已移除" -ForegroundColor Yellow
    }
    
    Write-Host "开机自启移除完成！" -ForegroundColor Yellow
}

function Show-Status {
    Write-Host "=== FunASR转录服务器状态 ===" -ForegroundColor Cyan
    
    # 检查注册表启动项
    $RegPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
    $RegEntry = Get-ItemProperty -Path $RegPath -Name $ServiceName -ErrorAction SilentlyContinue
    if ($RegEntry) {
        Write-Host "✓ 注册表启动项: 已设置" -ForegroundColor Green
        Write-Host "  路径: $($RegEntry.$ServiceName)" -ForegroundColor Gray
    } else {
        Write-Host "❌ 注册表启动项: 未设置" -ForegroundColor Red
    }
    
    # 检查任务计划
    $TaskInfo = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
    if ($TaskInfo) {
        Write-Host "✓ 任务计划: 已设置" -ForegroundColor Green
        Write-Host "  状态: $($TaskInfo.State)" -ForegroundColor Gray
    } else {
        Write-Host "❌ 任务计划: 未设置" -ForegroundColor Red
    }
    
    # 检查进程状态
    $Process = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*funasr*" }
    if ($Process) {
        Write-Host "✓ 服务进程: 正在运行 (PID: $($Process.Id))" -ForegroundColor Green
    } else {
        Write-Host "❌ 服务进程: 未运行" -ForegroundColor Red
    }
    
    Write-Host "=============================" -ForegroundColor Cyan
}

# 主逻辑
Write-Host "FunASR转录服务器开机自启管理工具" -ForegroundColor Cyan
Write-Host "项目位置: $ScriptPath" -ForegroundColor Gray
Write-Host ""

if (-not (Test-Administrator)) {
    Write-Host "❌ 此脚本需要管理员权限运行！" -ForegroundColor Red
    Write-Host "请右键点击PowerShell，选择'以管理员身份运行'" -ForegroundColor Yellow
    Read-Host "按回车键退出"
    exit 1
}

if ($Install) {
    Install-AutoStart
} elseif ($Uninstall) {
    Uninstall-AutoStart
} elseif ($Status) {
    Show-Status
} else {
    Write-Host "用法示例:" -ForegroundColor Yellow
    Write-Host "  安装开机自启: .\setup_autostart.ps1 -Install" -ForegroundColor White
    Write-Host "  移除开机自启: .\setup_autostart.ps1 -Uninstall" -ForegroundColor White
    Write-Host "  查看状态:     .\setup_autostart.ps1 -Status" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "请选择操作 [I]安装 / [U]移除 / [S]状态 / [Q]退出"
    switch ($choice.ToUpper()) {
        "I" { Install-AutoStart }
        "U" { Uninstall-AutoStart }
        "S" { Show-Status }
        "Q" { exit 0 }
        default { 
            Write-Host "无效选择，退出" -ForegroundColor Red
            exit 1
        }
    }
}

Read-Host "`n按回车键退出"