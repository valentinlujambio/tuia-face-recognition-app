# Despliegue local en Windows (PowerShell)

## Levantar todo

Entorno instalado y docker:

```powershell
.\scripts\start.ps1
```

Levanta Postgres (Docker), backend (http://localhost:8000/docs) y frontend (http://localhost:8080), y espera a que respondan.

Si los puertos 5432, 8000 u 8080 están ocupados, **el script baja lo que los use** (contenedor Docker, servicio de Windows, regla `netsh portproxy` o proceso) y lo imprime. Bajar un servicio de Windows o borrar una regla portproxy pide permisos de administrador (aparece un aviso de UAC). Los puertos se pueden cambiar con `-BackendPort` y `-FrontendPort`.

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
| `postgres hook exited with status 2` en la primera ejecución | El `psql` del `post_start` corre mientras Postgres inicializa por primera vez | Volver a correr `start.ps1` |
| `psycopg.OperationalError: ... autentificación password falló para el usuario "faces_user"` | Un Postgres nativo de Windows ocupa el 5432 y tapa al del contenedor | `start.ps1` baja ese servicio |
| `[winerror 10013] ... bind on address ('127.0.0.1', 8000)` | Una regla `netsh portproxy` (servicio `iphlpsvc`) ocupa el 8000 | `start.ps1` borra esa regla; imprime el comando para restaurarla |
| El 8000 u 8080 "no abre" | Otro contenedor Docker o proceso usa el puerto, o Gradio aún está arrancando (tarda) | `start.ps1` baja lo que use el puerto y espera hasta 2 min |
