# TP1 - Sistema de Reconocimiento Facial

Sistema de reconocimiento facial de punta a punta: dataset y preprocesamiento (Parte 1), comparación de una CNN propia contra InsightFace `buffalo_l` (Parte 2) y una app web (FastAPI + Gradio + PostgreSQL/pgvector) que registra identidades y reconoce personas en imágenes nuevas (Parte 3). La app usa InsightFace `buffalo_l`.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos (la que se entrega)
├── 2a-modelo-custom.ipynb       # Parte 2, auxiliar: entrenamiento del Modelo A (Colab)
├── 2b-modelo-preentrenado.ipynb # Parte 2, auxiliar: Modelo B y evaluación de un solo modelo
├── splits/                      # Salida de la Parte 1: splits.csv, discarded.csv, config.json
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── dataset/             # Parte 1: fuentes, calidad, duplicados, clustering, augmentation, splits
│       ├── vision/              # Detección (SCRFD), alineación, InputNormalization y OnnxFaceEmbedder
│       ├── models/              # Modelo A (CustomFaceCNN), cabezas softmax/ArcFace y export a ONNX
│       ├── training/            # Parte 2: datos, experimentos, entrenamiento y métricas por época
│       ├── evaluation/          # Parte 2: verificación, identificación, costo y contrato de embeddings
│       ├── services/
│       │   ├── face_service.py  # detect_faces, align_face y extract_embedding_from_face (Parte 3)
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── tests/                       # Tests unitarios de FaceService (pytest)
├── docs/                        # Capturas de la validación end-to-end y fotos de los integrantes
├── models/                      # Modelos .onnx, no se versionan (ver "Cómo correrlo")
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

## Cómo correrlo

La app usa el **Modelo B** (InsightFace `buffalo_l`): el reconocedor `w600k_r50.onnx` y el detector `det_10g.onnx`. Ninguno se versiona.

1. Descargar el paquete [`buffalo_l.zip`](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) de InsightFace (288 MB) y copiar `w600k_r50.onnx` y `det_10g.onnx` en `models/`:

   ```bash
   curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   unzip -j buffalo_l.zip w600k_r50.onnx det_10g.onnx -d models/
   ```

2. Levantar todo con Docker. `.env.docker.example` ya trae la configuración del modelo elegido (`MODEL_NAME=w600k_r50.onnx`, `DETECTOR_MODEL_NAME=det_10g.onnx`, `EMBEDDING_DIM=512`, `SIMILARITY_THRESHOLD=0.334`, `USE_PGVECTOR=true`):

   ```bash
   docker compose build
   docker compose up -d
   ```

3. Verificar: `curl http://localhost:8000/health` responde `{"status": "ok", "model": "w600k_r50.onnx"}`. El frontend queda en `http://localhost:8080`: registrar 3 a 5 personas con `/insert` (varias fotos frontales cada una) y probar `/predict` con fotos nuevas y con alguien no registrado, que tiene que salir como `unknown`.

**Sin Docker:** seguir la [Opción 2](#opción-2---configurando-el-ambiente-local) con `cp ../.env.local.example .env` desde `src/` y los dos `.onnx` en `src/models/`. Con `USE_PGVECTOR=false` no hace falta Postgres: los embeddings se guardan en `EMBEDDINGS_PATH`.

**Para usar el Modelo A** en lugar del B: bajar [`custom_cnn.onnx`](https://drive.google.com/file/d/1iWWYQ4wmxB1UdXM3lfTFq3anLCqIYvWS/view?usp=sharing) a `models/` y cambiar en el `.env` `MODEL_NAME=custom_cnn.onnx`, `EMBEDDING_DIM=128` y `SIMILARITY_THRESHOLD=0.780`. El detector sigue siendo `det_10g.onnx`. Al cambiar `EMBEDDING_DIM`, la tabla de pgvector se recrea y se borran los registros.

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`. Si su detector viene de una librería, sobrescriban `_load_detector` en `face_service.py`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado por el grupo (`src/lib/services/face_service.py`)

El pipeline de la app es el mismo de la Parte 1 (detector, alineación y normalización), con las clases de `src/lib/vision/`, así las caras que ve el modelo en la app son iguales a las de entrenamiento y evaluación.

1. **`detect_faces`:** SCRFD (`det_10g.onnx` de `buffalo_l`), cargado una sola vez en `_load_detector`. Antes de detectar agrega un borde negro de `DET_PAD_RATIO = 0.25` (SCRFD no encuentra caras que ocupan todo el cuadro; Parte 1, 2.4), filtra por `DET_SCORE_THRESHOLD` y `MIN_FACE_SIZE` y devuelve las cajas `(x1, y1, x2, y2)` ordenadas de mayor a menor área, recortadas a la imagen.
2. **`align_face`:** vuelve a detectar en un recorte con `ALIGN_MARGIN` de margen alrededor de la caja y toma la detección de mayor IoU con ella, para tener sus 5 landmarks. Con ellos alinea la imagen completa a la plantilla de 5 puntos de ArcFace (transformación de similaridad) a `FACE_SIZE × FACE_SIZE`. Devuelve los keypoints relativos a la caja, que el frontend dibuja. Si no encuentra landmarks, usa el recorte de la caja redimensionado y lo deja en el log.
3. **`extract_embedding_from_face`:** `InputNormalization` (BGR → RGB, `(x − 127,5) / 127,5`, NCHW) y el `.onnx` con onnxruntime (`OnnxFaceEmbedder`), o un `.pth` completo con torch; el embedding sale normalizado L2. Es el mismo código para los dos modelos: cambiar de modelo es cambiar el `.env`.

**Cambios en el código provisto, y por qué:**
- `_load_model` devuelve un `OnnxFaceEmbedder` en vez de una sesión cruda, para que la normalización de entrada sea la misma que en las notebooks.
- `_load_detector` carga SCRFD desde `models/<DETECTOR_MODEL_NAME>` (o, si está vacío, desde `~/.insightface`). El detector es un archivo aparte del reconocedor, como prevé el enunciado (3.5).
- `Settings` (`src/lib/config.py`) ignora las variables del `.env` que no son suyas. Antes, con `KAGGLE_API_TOKEN` en `src/.env` (lo usa la notebook 1), el backend no arrancaba.
- Variable nueva `DET_PAD_RATIO`, en los dos `.env.*.example`.
- `identify` usa `PgVectorEmbeddingStore.search` con pgvector y coseno: la base busca el registro más parecido con `<=>`. Con el store JSON o con `l2` sigue recorriendo todos. Para eso se corrigió `search`, que fallaba al ejecutarse.

**Consistencia app ↔ notebook:** la misma foto procesada por la app y por la Parte 1 da embeddings con coseno entre 0,991 y 0,998. La diferencia viene de detectar sobre la imagen original o sobre el recorte.

### Validación end-to-end

Se probó el stack completo (Docker, `w600k_r50.onnx`, `SIMILARITY_THRESHOLD=0.334`, `USE_PGVECTOR=true`) con identidades del dataset y con fotos de los integrantes (usadas con su consentimiento, en `docs/fotos_*`). Las capturas del frontend están en [`docs/capturas/`](docs/capturas/).

| Caso | Resultado |
|---|---|
| Registro de Chris Evans, Zendaya y Tom Holland (3 fotos cada una) y de Valentín Lujambio (3 fotos) | los 12 registros terminan en `done` |
| Una persona registrada, con una foto no usada en el registro | etiqueta correcta, score 0,57 a 0,94 ([captura](docs/capturas/predict_lucas_reconocido.jpg)) |
| Varias personas en una imagen | cada rostro con su etiqueta y sus keypoints ([captura](docs/capturas/predict_grupo_cajas_keypoints.jpg)) |
| Persona no registrada | `unknown`, score 0,06 a 0,16 ([captura](docs/capturas/unknown_lucas_sin_registrar.jpg)) |
| Imagen sin rostros | `detections: []`, estado `done` |
| Registro con 2 rostros | estado `failed`: "Exactly one face must be detected for identity registration." |
| Registro de una identidad desde el frontend | recorte alineado y estado "completado (registro)" ([captura](docs/capturas/registro_lucas_ramirez.jpg)) |

Con una sola foto de registro, Lucas Ramirez pasó de `unknown` (0,16) a ser reconocido con 0,83 en una foto distinta. Las fotos propias no son parte del dataset ni de la evaluación de la Parte 2.

## Modelos comparados (Parte 2)

Detalle completo en [`2-train.ipynb`](2-train.ipynb). Los dos modelos se evalúan con los mismos pares, la misma galería y las mismas consultas, y el costo se mide en la misma máquina y la misma corrida.

| Métrica | Modelo A: CNN custom | Modelo B: InsightFace `buffalo_l` |
|---|---|---|
| AUC (verificación, `test_open`) | 0,899 | 0,99996 |
| EER | 18,2 % | 0,36 % |
| TAR @ FAR = 1e-3 | 15,3 % | 99,5 % |
| Top-1 / Top-5 (sin umbral) | 86,1 % / 96,5 % | 100 % / 100 % |
| Precision / Recall / F1 (macro, con umbral) | 0,955 / 0,387 / 0,525 | 0,994 / 0,999 / 0,996 |
| Conocidos bien identificados / desconocidos rechazados | 39,0 % / 98,9 % | 99,9 % / 99,2 % |
| Umbral (1:N, FPIR = 1 % en validación) | 0,780 | 0,334 |
| Tiempo de inferencia en CPU (ms/rostro, de a uno) | 4,3 | 39 |
| Tamaño / parámetros / dimensión | 10,8 MB / 2,8 M / 128 | 166 MB / 43,6 M / 512 |

**Modelo elegido: B, InsightFace `buffalo_l` (`w600k_r50.onnx`).** Con el umbral que hace falta para rechazar desconocidos, el Modelo A reconoce a menos de la mitad de las personas registradas; el B, al 99,9 %. El A es 15 veces más chico y 9 veces más rápido, pero los 39 ms del B alcanzan de sobra para una app que procesa una imagen por pedido. La contra del B es la licencia: los pesos son solo para uso no comercial / investigación, válidos para este trabajo académico. Las métricas del B son optimistas: las celebridades de Pins probablemente estén en su entrenamiento (WebFace600K), y ArcFace se usó para curar el dataset.

**Umbral configurado: `SIMILARITY_THRESHOLD=0.334` (`SIMILARITY_METRIC=cosine`).** Se calibró en validación, no a ojo, simulando la app: una galería de 5 fotos por persona de `val_closed` (80 personas, 400 registros), el resto de `val_closed` como consultas conocidas y `val_open` como desconocidos. Es el umbral con el que se acepta solo al 1 % de los desconocidos (FPIR = 1 %), y conserva el 99,8 % de los conocidos. Los umbrales de verificación 1:1 (EER 0,229, FAR = 1e-3 0,237) aceptaban como alguien registrado al 31–38 % de los desconocidos, porque la app compara cada cara contra todos los registros.

**Análisis de errores:** en el B, los falsos negativos son sobre todo perfiles, oclusiones y baja resolución, y se concentran en Taylor Swift. Los pocos falsos positivos son Emilia Clarke contra Jessica Barden, de rasgos muy parecidos. El A confunde personas del mismo género, color de pelo y expresión, y falla con oclusiones, poca luz y perfiles. La razón principal es que se entrenó con 9.006 imágenes de 80 personas, contra ~12 millones de 600 mil del B.

### Modelo A: CNN custom

- **Descarga (.onnx, público, solo lectura):** [custom_cnn.onnx](https://drive.google.com/file/d/1iWWYQ4wmxB1UdXM3lfTFq3anLCqIYvWS/view?usp=sharing). Copiarlo en `models/`.
- **Notebook en Colab (ejecutada):** [2a-modelo-custom](https://colab.research.google.com/drive/1jsKVLagYVW_ujwkjycdLnifEWf6rrMGf?usp=sharing)
- **Arquitectura:** ResNet chica de 4 etapas (2 bloques residuales cada una, BatchNorm, PReLU, global average pooling), ~2,8M de parámetros, código en `src/lib/models/custom_cnn.py`.
- **Entrenamiento:** softmax + SGD (exp 1) y luego fine-tuning con ArcFace (s = 30, m = 0,5) y lr 0,001 constante, inicializado desde ese checkpoint (exp 5). Detalle de los 6 experimentos en la notebook.
- **Resultado en val_open (identidades no vistas):** AUC 0,905, EER 0,176.
- **Configuración en `.env`:** `MODEL_NAME=custom_cnn.onnx`, `EMBEDDING_DIM=128`, `FACE_SIZE=112`, `SIMILARITY_THRESHOLD=0.780`. Preprocesamiento: RGB, `(x - 127.5) / 127.5`, NCHW, y normalización L2 del embedding.

### Modelo B: InsightFace `buffalo_l` (elegido)

- **Descarga:** paquete [`buffalo_l.zip`](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) de InsightFace; se usan `w600k_r50.onnx` (reconocedor) y `det_10g.onnx` (detector), en `models/`.
- **Reconocedor:** IResNet-50 entrenada con ArcFace sobre WebFace600K (~600 mil identidades, ~12 millones de imágenes), embedding de 512 dimensiones. Se usa sin reentrenar.
- **Detector:** SCRFD-10GF, entrenado con WIDER FACE; da la caja y 5 landmarks.
- **Licencia:** código MIT; pesos solo para uso no comercial / investigación.
- **Configuración en `.env`:** `MODEL_NAME=w600k_r50.onnx`, `DETECTOR_MODEL_NAME=det_10g.onnx`, `EMBEDDING_DIM=512`, `FACE_SIZE=112`, `SIMILARITY_THRESHOLD=0.334`.

## Dataset

Detalle en [`1-pre-processing.ipynb`](1-pre-processing.ipynb).

- **Fuente:** [Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) (Kaggle, `hereisburak/pins-face-recognition`): 105 celebridades y 17.534 imágenes de Pinterest. Licencia CC0 declarada en Kaggle; los derechos de cada foto siguen siendo de sus autores. No se usaron fotos propias. Las imágenes no se versionan: solo `splits/` (rutas, etiquetas y parámetros).
- **Calidad y filtrado:** se descartaron 671 imágenes (3,8 %): 580 duplicados (coseno de ArcFace ≥ 0,95), 61 por exposición extrema, 13 borrosas (varianza del Laplaciano < 30), 6 sin cara o con varias caras sin una dominante, 8 outliers de identidad y 3 errores de etiqueta. Quedan **16.863 imágenes de 105 personas**.
- **Balance:** entre 86 (Lionel Messi) y 226 (Robert Downey Jr) imágenes por persona, media 161: desbalance moderado de 2,6×.
- **Variaciones:** mayormente fotos frontales y bien iluminadas, con una cola de perfiles y caras inclinadas, ~7 % en blanco y negro, maquillaje y producción profesional.
- **Detección y alineación:** SCRFD con borde de 25 %, alineación de 5 puntos a 112×112.
- **Splits:** *closed-set* con 80 personas (9.006 `train`, 1.980 `val_closed`, 1.974 `test_closed`) y *open-set* con identidades enteras que el Modelo A nunca ve (12 personas y 2.003 imágenes en `val_open`, 13 personas y 1.900 imágenes en `test_open`). Las copias de una misma foto quedan siempre en el mismo split.
- **Augmentation** (solo en train): flip, rotación y traslación, brillo y contraste, color, escala de grises, blur o ruido, oclusión y compresión.

## Tests

Tests unitarios de `FaceService` en [`tests/test_face_service.py`](tests/test_face_service.py): `detect_faces` (una cara, dos caras, sin caras), `align_face` (imagen `FACE_SIZE`², keypoints `(5, 2)` relativos a la caja y en orden InsightFace, caso sin landmarks), `extract_embedding_from_face` (dimensión 512, `float` de Python, norma L2 = 1, misma persona más cerca que distinta y separadas por el umbral calibrado), `identify`, `register_identity` (incluye 2 rostros, sin rostros e imagen ilegible) y `predict` (reconocida, `unknown`, sin rostros y varias personas).

Usan los modelos reales (`w600k_r50.onnx` y `det_10g.onnx`, en `models/` o en `~/.insightface/models/buffalo_l`) y las fotos de `docs/fotos_*`; si faltan los modelos, los tests se saltean. No necesitan Postgres: usan un store en memoria.

```bash
pytest          # desde la raíz del repo; `pytest.ini` ya agrega `src/` al path
```

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
