# Levanta Postgres (Docker), backend y frontend, cada uno en su ventana, y espera a que respondan.
# Si un puerto esta ocupado, BAJA lo que lo use (contenedor Docker, servicio de Windows o proceso).
# Requiere el entorno ya instalado (scripts/setup_env.ps1) y src/.env creado.
# Uso (desde TP_1):  .\scripts\start.ps1 [-BackendPort 8000] [-FrontendPort 8080]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 8080
)
$root = Split-Path $PSScriptRoot -Parent
$src  = Join-Path $root "src"
$venv = Join-Path $root ".venv\Scripts"
$PostgresPort = 5432
Set-Location $root

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
# Procesos que solo reenvian puertos de Docker/WSL: no se matan, se baja el contenedor que los publica
$relays = "com.docker.backend", "wslrelay", "svchost", "System", "vpnkit"

function Invoke-Admin([string]$Command) {
    if ($isAdmin) { Invoke-Expression $Command }
    else { Start-Process powershell -Verb RunAs -Wait -ArgumentList "-NoProfile", "-Command", $Command }
}

function Get-OwningService([int]$ProcessId) {
    for ($i = 0; $i -lt 5 -and $ProcessId; $i++) {
        $svc = Get-CimInstance Win32_Service -Filter "ProcessId=$ProcessId" | Select-Object -First 1
        if ($svc) { return $svc }
        $ProcessId = (Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId").ParentProcessId
    }
}

function Free-Port([int]$Port, [string]$KeepContainerId = "") {
    foreach ($line in docker ps --filter "publish=$Port" --format "{{.ID}} {{.Names}}") {
        $id, $name = $line -split ' ', 2
        if ($KeepContainerId -and $KeepContainerId.StartsWith($id)) { continue }
        Write-Host "Puerto ${Port}: bajando contenedor '$name'"
        docker stop $id | Out-Null
    }
    # Regla de netsh portproxy (la atiende svchost/iphlpsvc): se borra solo la del puerto ocupado
    foreach ($l in netsh interface portproxy show v4tov4) {
        if ($l -match "^\s*(\S+)\s+$Port\s+(\S+)\s+(\d+)\s*$") {
            Write-Host "Puerto ${Port}: borrando regla portproxy $($Matches[1]):$Port -> $($Matches[2]):$($Matches[3])"
            Write-Host "  Para restaurarla: netsh interface portproxy add v4tov4 listenaddress=$($Matches[1]) listenport=$Port connectaddress=$($Matches[2]) connectport=$($Matches[3])"
            Invoke-Admin "netsh interface portproxy delete v4tov4 listenaddress=$($Matches[1]) listenport=$Port"
        }
    }
    $pids = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids) {
        $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
        if (-not $proc -or $relays -contains $proc.ProcessName) { continue }
        $svc = Get-OwningService $p
        if ($svc) {
            Write-Host "Puerto ${Port}: bajando servicio de Windows '$($svc.Name)'"
            Invoke-Admin "Stop-Service -Name '$($svc.Name)' -Force"
        } else {
            Write-Host "Puerto ${Port}: matando proceso '$($proc.ProcessName)' ($p)"
            Stop-Process -Id $p -Force
        }
    }
    Start-Sleep 2
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Where-Object { $relays -notcontains (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName }) {
        Write-Error "No se pudo liberar el puerto $Port."; exit 1
    }
}

# Chequeo previo: sin modelo el backend no arranca
$modelName = (Get-Content (Join-Path $src ".env") | Where-Object { $_ -match '^MODEL_NAME=(.+)$' } | ForEach-Object { $Matches[1].Trim() } | Select-Object -Last 1)
$modelFile = Join-Path $src "models\$modelName"
if (-not (Test-Path $modelFile)) { Write-Error "Falta el modelo: $modelFile (el backend no arranca sin el)."; exit 1 }

# Liberar puertos (en el 5432 se respeta el Postgres de este proyecto)
$ourPg = docker compose ps -q postgres 2>$null
Free-Port $PostgresPort $ourPg
Free-Port $BackendPort
Free-Port $FrontendPort

docker compose up postgres -d --wait
if ($LASTEXITCODE -ne 0) { Write-Error "Postgres no levanto: revisar que Docker Desktop este corriendo. En la primera ejecucion puede fallar el hook post_start; reintentar."; exit 1 }

Start-Process powershell -WorkingDirectory $src -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='backend :$BackendPort'; & '$venv\uvicorn.exe' app.main:app --reload --port $BackendPort"
Start-Process powershell -WorkingDirectory $src -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='frontend :$FrontendPort'; `$env:BACKEND_URL='http://127.0.0.1:$BackendPort'; & '$venv\uvicorn.exe' frontend.app:app --port $FrontendPort"

# El primer arranque tarda (torch, gradio): esperar hasta 2 minutos a cada servicio
$targets = @(
    @{ Name = "Backend";  Url = "http://127.0.0.1:$BackendPort/docs" },
    @{ Name = "Frontend"; Url = "http://127.0.0.1:$FrontendPort/" }
)
foreach ($t in $targets) {
    $ok = $false
    for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
        try { $ok = (Invoke-WebRequest $t.Url -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { Start-Sleep 2 }
    }
    if ($ok) { Write-Host "$($t.Name): $($t.Url)  OK" } else { Write-Warning "$($t.Name) no respondio en $($t.Url): mirar su ventana." }
}
Write-Host "Cerrar las ventanas para detener backend/frontend; 'docker compose stop postgres' para la base."
