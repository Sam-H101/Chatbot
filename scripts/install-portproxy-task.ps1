<#
.SYNOPSIS
    Registers (or removes) the scheduled task that runs portproxy.ps1 at startup and at logon.

.DESCRIPTION
    Copies portproxy.ps1 into %ProgramData%\Chatbot, where only administrators can change it (the
    task runs it with full rights, so it must not run a copy anyone can edit: if someone else made
    that folder first, Administrators take it over and its permissions are reset, and the install
    stops if that didn't work), and registers the task "Chatbot portproxy" to run that copy as
    you, with highest privileges:
      - at startup (30 s after boot), whether you are logged on or not (no password is stored), and
      - at logon.
    Each run starts WSL if it isn't running and keeps it running (the Docker containers start
    again by themselves), then points the port proxy at WSL's current address. The task is run
    once at the end, and its log is shown.

    Run it again after changing portproxy.ps1, or to change -Distro or -Port. -Remove deletes the
    task and the copy; the port proxy and firewall rules stay (the end of the run says how to
    remove them).

    Needs an elevated PowerShell (Run as administrator). -WhatIf shows what it would do, without
    elevation.

.PARAMETER Interactive
    Run only while you are logged on, at logon. Use it if the log shows WSL failing to start from
    the startup run (Windows may not let WSL start without anyone logged on).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\install-portproxy-task.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Remove,
    [switch]$Interactive,
    [string]$TaskName = 'Chatbot portproxy',
    [string]$Distro = 'vostok',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$User = "$env:USERDOMAIN\$env:USERNAME"
)

$ErrorActionPreference = 'Stop'
$dry = [bool]$WhatIfPreference
# Load the module before -WhatIf applies: loading it under -WhatIf prints a line per alias.
$WhatIfPreference = $false
Import-Module ScheduledTasks
$WhatIfPreference = $dry
$dir = Join-Path $env:ProgramData 'Chatbot'
$target = Join-Path $dir 'portproxy.ps1'
$log = Join-Path $dir 'portproxy.log'
$source = Join-Path $PSScriptRoot 'portproxy.ps1'

function Install-Copy([string]$Source, [string]$Dir, [string]$Target) {
    # Copies $Source to $Target in a folder only Administrators and SYSTEM can change. Users may create
    # folders in ProgramData, and whoever made this one (or a file in it) first would own it and could
    # change it whatever its permissions say; so Administrators take it over first. SIDs, not names, so
    # it works in any Windows language: Administrators S-1-5-32-544, SYSTEM S-1-5-18, Users S-1-5-32-545.
    if ((Test-Path -LiteralPath $Dir) -and
        ((Get-Item -LiteralPath $Dir -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "$Dir is a link, not a folder: delete it, then run this again"
    }
    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    & takeown.exe /F $Dir /A | Out-Null  # takeown can take it from anyone; icacls /setowner can't
    if ($LASTEXITCODE -ne 0) { throw "takeown failed on $Dir (exit code $LASTEXITCODE)" }
    & icacls.exe $Dir /reset /Q | Out-Null  # drops permissions set on it before
    & icacls.exe $Dir /inheritance:r /grant:r '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-545:(OI)(CI)RX' /Q | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "icacls failed on $Dir (exit code $LASTEXITCODE)" }
    # A new file, which takes the folder's permissions (one already there keeps its own), owned by
    # Administrators whatever this account's default owner for new files is.
    if (Test-Path -LiteralPath $Target) {
        & takeown.exe /F $Target /A | Out-Null  # its permissions may keep administrators from deleting it
        & icacls.exe $Target /reset /Q | Out-Null
        Remove-Item -LiteralPath $Target -Force
    }
    Copy-Item -LiteralPath $Source -Destination $Target
    & takeown.exe /F $Target /A | Out-Null
    # Check it took: no one else may own them, or have any right to change them.
    $admins = 'S-1-5-32-544', 'S-1-5-18'
    $write = 0x2 -bor 0x4 -bor 0x10 -bor 0x40 -bor 0x100 -bor 0x10000 -bor 0x40000 -bor 0x80000 -bor 0x10000000 -bor 0x40000000
    foreach ($path in $Dir, $Target) {
        $acl = Get-Acl -LiteralPath $path
        $owner = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
        $others = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]) | Where-Object {
                "$($_.AccessControlType)" -eq 'Allow' -and ([int64]$_.FileSystemRights -band $write) -and
                $admins -notcontains $_.IdentityReference.Value } | ForEach-Object { $_.IdentityReference.Value })
        if ($admins -notcontains $owner -or $others) {
            throw ("$path is not locked down (owner $owner" + $(if ($others) { "; may be changed by $($others -join ', ')" }) +
                '): the task was not registered')
        }
    }
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$elevated = (New-Object Security.Principal.WindowsPrincipal($identity)).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elevated -and -not $dry) {
    Write-Host ('Registering the task needs an elevated PowerShell (Run as administrator). ' +
        'Nothing was changed. Add -WhatIf to see what it would do.')
    exit 1
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($Remove) {
    if ($existing -and $PSCmdlet.ShouldProcess("scheduled task '$TaskName'", 'Unregister')) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed the scheduled task '$TaskName'."
    } elseif (-not $existing) {
        Write-Host "There is no scheduled task '$TaskName'."
    }
    if ((Test-Path $target) -and $PSCmdlet.ShouldProcess($target, 'Delete')) {
        Remove-Item -Path $target -Force
        Write-Host "Deleted $target (its log, $log, is kept)."
    }
    Write-Host ''
    Write-Host 'The port proxy and firewall rules are left as they are. To close the port to the LAN too:'
    Write-Host "  netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0"
    Write-Host "  netsh advfirewall firewall delete rule name=`"Chatbot $Port`""
    exit 0
}

if (-not (Test-Path $source)) {
    Write-Host "Can't find $source (keep this script next to portproxy.ps1)."
    exit 1
}

# 1. The copy the task runs, in a folder only administrators (and SYSTEM) can change.
if ($PSCmdlet.ShouldProcess("$source -> $target (in $dir, owned by Administrators, which users may only read)",
        'Restrict the folder and copy')) {
    Install-Copy -Source $source -Dir $dir -Target $target
}

# 2. The task.
$powershell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$target`" -Distro $Distro -Port $Port"
$action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User $User
if ($Interactive) {
    $triggers = @($atLogon)
    $principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Highest
    $when = "at logon of $User"
} else {
    $atStartup = New-ScheduledTaskTrigger -AtStartup
    $atStartup.Delay = 'PT30S'  # let the network and the WSL service come up first
    $triggers = @($atStartup, $atLogon)
    # S4U: runs whether you are logged on or not, without storing your password.
    $principal = New-ScheduledTaskPrincipal -UserId $User -LogonType S4U -RunLevel Highest
    $when = "at startup (after 30 s) and at logon of $User"
}
# The script takes seconds (a minute or so while WSL boots); the hidden wsl.exe it starts outlives it.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
$description = "Starts WSL distro '$Distro' and keeps it running, then points the port proxy 0.0.0.0:$Port at its " +
    "current address (scripts\portproxy.ps1 of the chatbot repo). Log: $log"

Write-Host "Task '$TaskName': $when, with highest privileges"
Write-Host "  runs: $powershell $arguments"
if ($PSCmdlet.ShouldProcess("scheduled task '$TaskName'", $(if ($existing) { 'Replace' } else { 'Register' }))) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Principal $principal `
        -Settings $settings -Description $description -Force | Out-Null
    Write-Host "Registered the scheduled task '$TaskName'."
}

# 3. Run it once now, and show what it did.
if ($PSCmdlet.ShouldProcess("scheduled task '$TaskName'", 'Run once now')) {
    $since = if (Test-Path $log) { @(Get-Content -Path $log).Count } else { 0 }
    $started = (Get-Date).AddSeconds(-5)  # LastRunTime has whole seconds
    Start-ScheduledTask -TaskName $TaskName
    $deadline = (Get-Date).AddSeconds(180)
    do {
        Start-Sleep -Seconds 2
        $state = (Get-ScheduledTask -TaskName $TaskName).State
        $info = Get-ScheduledTaskInfo -TaskName $TaskName
    } while ((Get-Date) -lt $deadline -and ($state -in 'Running', 'Queued' -or $info.LastRunTime -lt $started))
    $result = $info.LastTaskResult
    Write-Host "The task ran (result $result; 0 means it worked). Its log, $log :"
    if (Test-Path $log) { Get-Content -Path $log | Select-Object -Skip $since | ForEach-Object { Write-Host "  $_" } }
    if ($result -ne 0) {
        Write-Host ('It did not work. If the log shows WSL failing to start, run this again with -Interactive ' +
            '(the task then runs at logon only, in your session).')
        exit 1
    }
}
exit 0
