# Despliegue local en Windows (PowerShell)

## Levantar todo

Entorno instalado y docker:

```powershell
.\scripts\start.ps1
```

Levanta Postgres (Docker), backend (http://localhost:8000/docs) y frontend (http://localhost:8080).

## Instalación (una sola vez)

```powershell
.\scripts\setup_env.ps1                  # venv 3.12 + dependencias + insightface sin compilar
Copy-Item .env.local.example src\.env    # config del backend
```

Además hay que dejar el modelo entrenado en `src\models\<MODEL_NAME>` . Sin él el backend no arranca (`ValueError: Model path does not exist: models\face_detection.pth`); el frontend sí levanta solo.

## Problemas y como lo resolvimos

| Problema | Causa | Solución |
|---|---|---|
| `tifffile==2026.4.11 ... Python>=3.12` | El venv se creó con Python 3.11 | `uv venv --python 3.12 .venv` |
| `source : El término 'source' no se reconoce` | `source` es de bash; en Windows el script está en `Scripts`, no en `bin` | `.venv\Scripts\Activate.ps1` |
| `Failed to build insightface==0.7.3` / `Microsoft Visual C++ 14.0 or greater is required` | No hay wheel para Windows y compila una extensión C++ (`mesh_core_cython`, render 3D) | Instalar desde el sdist con `setup.py` parcheado sin esa extensión (`--no-deps --no-build-isolation`) |
| `error: could not create ... El nombre del archivo o la extensión es demasiado largo` | Al buildear insightface desde una ruta larga se pasa el límite de 260 caracteres de Windows | Buildear desde una ruta corta (`%TEMP%\ifb`) |
| `ImportError: cannot import name 'mesh_core_cython'` al hacer `import insightface` | `insightface` importa el módulo compilado aunque no se use | Stub `mesh_core_cython.py` en `site-packages\insightface\thirdparty\face3d\mesh\cython\` |
| `uvloop does not support Windows` | `uvloop` solo existe para Linux/Mac | Se excluye del install; `uvicorn` usa `asyncio` |
