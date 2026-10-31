# Token de Kaggle para bajar el dataset

Para ejecutar notebook tenemos que buscar una api key de Kaggle

## Pasos

1. En kaggle.com: perfil → **Settings** → sección **API** → **Create New Token**. Kaggle muestra un token `KGAT_...` (ya no descarga un `kaggle.json`).

2. Guardarlo en `C:\Users\<usuario>\.kaggle\access_token`
   
Alternativa : completar `KAGGLE_API_TOKEN=` en `src\.env` (la notebook lo carga antes de descargar). Nunca en `.env.local.example`, que se versiona.

3. Instalar las dependencias de la notebook en el venv:

```powershell
pip install ipykernel kagglehub
```


