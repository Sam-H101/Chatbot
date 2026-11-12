<#
.SYNOPSIS
    Points the chatbot's LAN port at WSL's current address.

.DESCRIPTION
    The app runs in Docker inside a WSL distro, and other machines reach it through a Windows
    port proxy rule (listen 0.0.0.0:8000, connect <WSL address>:8000) plus an inbound firewall
    rule. WSL gets a new address when it restarts, which leaves the rule pointing nowhere.
    (WSL's mirrored networking would avoid the proxy, but it needs Windows 11 22H2 or later.)

    This script:
      1. starts WSL if it isn't running, and keeps it running with a hidden
         `wsl.exe -d <distro> --exec sleep infinity` (WSL stops an idle distro after a few
         seconds otherwise; its Docker containers start again by themselves);
      2. reads the distro's current address (`hostname -I`, the first one);
      3. adds or updates the v4tov4 port proxy rule for the listen address and port, if it
         doesn't point there already;
      4. makes sure the IP Helper service (which carries the port proxy) is running and the
         firewall rule exists.
    Running it again changes nothing when all is in place. Every run is logged to -LogPath.

    Changing anything needs an elevated PowerShell (Run as administrator). Without elevation,
    or with -DryRun or -WhatIf, it only prints what it would do (reading WSL's address still
    starts WSL if it isn't running). install-portproxy-task.ps1 runs it at startup and logon.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\portproxy.ps1 -DryRun
    Shows what it would do; works without elevation.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\portproxy.ps1
    From an elevated PowerShell: points the rule at WSL now.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Distro = 'vostok',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$ListenAddress = '0.0.0.0',
    [string]$RuleName = "Chatbot $Port",
    [string]$LogPath = (Join-Path $env:ProgramData 'Chatbot\portproxy.log'),
    [switch]$DryRun,
    [switch]$NoKeepAlive
)

$dry = $DryRun -or [bool]$WhatIfPreference
# Load the modules used below before -WhatIf applies: loading one under it prints a
# "What if" line for every alias the module defines.
$WhatIfPreference = $false
Import-Module CimCmdlets, NetSecurity -ErrorAction SilentlyContinue
$WhatIfPreference = $dry
$wsl = Join-Path $env:WINDIR 'System32\wsl.exe'

function Write-Log([string]$Message) {
    $line = '{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $Message
    Write-Host $line
    if ($dry) { return }  # a dry run only prints
    try {
        $dir = Split-Path -Parent $LogPath
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -WhatIf:$false | Out-Null }
        if ((Test-Path $LogPath) -and (Get-Item $LogPath).Length -gt 1MB) {
            Move-Item -Path $LogPath -Destination "$LogPath.old" -Force -WhatIf:$false
        }
        Add-Content -Path $LogPath -Value $line -Encoding UTF8 -WhatIf:$false
    } catch {
        Write-Host "(could not write the log $LogPath : $_)"
    }
}

function Get-KeepAlive {
    # The wsl.exe processes already keeping the distro running (started by this script, or by hand).
    $pattern = '(-d|--distribution)\s+"?' + [regex]::Escape($Distro) + '"?\s.*\bsleep\s+infinity'
    Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match $pattern }
}

function Get-WslAddress {
    # `wsl -d <distro>` boots the distro if it isn't running. Its first address is eth0's.
    $ErrorActionPreference = 'Continue'
    for ($try = 1; $try -le 6; $try++) {
        $out = & $wsl -d $Distro -- hostname -I 2>&1
        $code = $LASTEXITCODE
        $text = (($out | Out-String) -replace "`0", '' -replace '\s+', ' ').Trim()  # wsl.exe's own messages are UTF-16
        $ip = $text -split ' ' | Where-Object { $_ -match '^\d{1,3}(\.\d{1,3}){3}$' } | Select-Object -First 1
        if ($code -eq 0 -and $ip) { return $ip }
        Write-Log "No address from WSL distro '$Distro' yet (try $try of 6, exit code $code): $text"
        if ($text -match 'DISTRO_NOT_FOUND') { break }  # waiting won't help
        if ($try -lt 6) { Start-Sleep -Seconds 10 }
    }
    return $null
}

function Get-ProxyTarget {
    # The rule as netsh stores it: value "<listen address>/<port>" = "<connect address>/<port>".
    $key = 'HKLM:\SYSTEM\CurrentControlSet\Services\PortProxy\v4tov4\tcp'
    if (-not (Test-Path $key)) { return $null }
    (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue)."$ListenAddress/$Port"
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$elevated = (New-Object Security.Principal.WindowsPrincipal($identity)).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elevated -and -not $dry) {
    Write-Host ('Changing the port proxy and the firewall needs an elevated PowerShell (Run as administrator). ' +
        'Nothing was changed. Add -DryRun to see what it would do.')
    exit 1
}

Write-Log ("portproxy.ps1: distro '$Distro', ${ListenAddress}:$Port, firewall rule '$RuleName'" +
    $(if ($dry) { ' (dry run: nothing is changed)' } else { '' }))

# 1. Start WSL, and keep it running.
if (-not $NoKeepAlive) {
    $keep = @(Get-KeepAlive)
    if ($keep.Count -gt 0) {
        Write-Log "WSL is kept running already (wsl.exe process $($keep[0].ProcessId))"
    } elseif ($PSCmdlet.ShouldProcess("$wsl -d $Distro --exec sleep infinity", 'Start, hidden, to keep WSL running')) {
        Start-Process -FilePath $wsl -ArgumentList @('-d', $Distro, '--exec', 'sleep', 'infinity') -WindowStyle Hidden
        Write-Log "Started '$wsl -d $Distro --exec sleep infinity' to keep WSL running"
    }
}

# 2. WSL's address.
$ip = Get-WslAddress
if (-not $ip) {
    Write-Log "Gave up: WSL distro '$Distro' has no address. Nothing was changed."
    exit 1
}
Write-Log "WSL distro '$Distro' is at $ip"

# 3. The port proxy rule.
$want = "$ip/$Port"
$current = Get-ProxyTarget
if ($current -eq $want) {
    Write-Log "Port proxy ${ListenAddress}:$Port already points at ${ip}:$Port"
} else {
    $verb = if ($current) { 'set' } else { 'add' }  # set replaces the rule for this listen address and port
    $netsh = @('interface', 'portproxy', $verb, 'v4tov4', "listenaddress=$ListenAddress", "listenport=$Port",
        "connectaddress=$ip", "connectport=$Port")
    if ($PSCmdlet.ShouldProcess("netsh $($netsh -join ' ')", 'Run')) {
        $out = & netsh.exe @netsh 2>&1
        $now = Get-ProxyTarget
        if ($LASTEXITCODE -ne 0 -or $now -ne $want) {
            Write-Log "netsh failed (exit code $LASTEXITCODE, rule now '$now'): $(($out | Out-String).Trim())"
            exit 1
        }
        Write-Log ("Port proxy ${ListenAddress}:$Port now points at ${ip}:$Port (was " +
            $(if ($current) { $current -replace '/', ':' } else { 'not set' }) + ')')
    }
}

# 4. The IP Helper service and the firewall rule.
$svc = Get-Service -Name iphlpsvc -ErrorAction SilentlyContinue
if (-not $svc) {
    Write-Log 'The IP Helper service (iphlpsvc) is missing: the port proxy will not work'
} elseif ($svc.Status -ne 'Running') {
    if ($PSCmdlet.ShouldProcess('IP Helper service (iphlpsvc), which carries the port proxy', 'Start')) {
        Start-Service -Name iphlpsvc
        Write-Log 'Started the IP Helper service (iphlpsvc)'
    }
}

$rules = @(Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)
if ($rules.Count -eq 0) {
    # Profile Any, like the rule set up by hand before this script: the LAN may be classed Public.
    $cmd = "New-NetFirewallRule -DisplayName '$RuleName' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile Any"
    if ($PSCmdlet.ShouldProcess($cmd, 'Run')) {
        New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
            -Profile Any -ErrorAction Stop | Out-Null
        Write-Log "Added firewall rule '$RuleName' (inbound TCP $Port)"
    }
} elseif (-not ($rules | Where-Object { "$($_.Enabled)" -eq 'True' })) {
    Write-Log "Firewall rule '$RuleName' exists but is disabled: left as it is, so the LAN can't reach port $Port"
} else {
    Write-Log "Firewall rule '$RuleName' is in place"
}
exit 0
