# Levanta Postgres (Docker), backend (:8000) y frontend (:8080), cada uno en su ventana.
# Requiere el entorno ya instalado (scripts/setup_env.ps1) y src/.env creado.
# Uso (desde TP_1):  .\scripts\start.ps1
$root = Split-Path $PSScriptRoot -Parent
$src  = Join-Path $root "src"
$venv = Join-Path $root ".venv\Scripts"

Set-Location $root
docker compose up postgres -d --wait
if ($LASTEXITCODE -ne 0) { Write-Error "Postgres no levanto: revisar que Docker Desktop este corriendo."; exit 1 }

Start-Process powershell -WorkingDirectory $src -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='backend :8000'; & '$venv\uvicorn.exe' app.main:app --reload --port 8000"
Start-Process powershell -WorkingDirectory $src -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='frontend :8080'; & '$venv\uvicorn.exe' frontend.app:app --port 8080"

Write-Host "Backend:  http://localhost:8000/docs"
Write-Host "Frontend: http://localhost:8080"
Write-Host "Cerrar las ventanas para detener backend/frontend; 'docker compose stop postgres' para la base."
