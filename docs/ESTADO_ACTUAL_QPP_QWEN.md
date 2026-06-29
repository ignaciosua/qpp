# Estado actual: QPP / BitTrit sobre Qwen

Fecha: 2026-06-28

Este documento fija el punto exacto donde estamos parados con el enfoque QPP para Qwen.

## Resumen corto

Ya logramos pasar de una compresion teorica/falsa a una compresion real parcial.

El primer intento QPP usaba `argsort` por fila. Eso hacia que los parametros de curva fueran pocos, pero obligaba a guardar una permutacion por cada fila. Al contar esa metadata, la compresion real caia a aproximadamente `1x`.

El enfoque actual usa:

- orden compartido por bloque de filas;
- anchors QPP por fila;
- calibracion por activaciones;
- seleccion greedy de modulos;
- conteo real de bytes;
- ejecucion real con modulos `QPPCompressedLinear`.

Esto ya ejecuta Qwen con algunos modulos almacenados en formato comprimido. Con `anchors=32` ya paso el gate de calidad parcial para `attention8`: Delta PPL `+0.2208` en 32768 tokens con compresion runtime `22.56x` en los modulos aceptados. Ademas, al comprimir `q/k/v` con gate global `Delta PPL <= 0.5`, ya se observa una reduccion real de bytes persistentes del modelo de aproximadamente `-8.995 MB`. Todavia no se demuestra reduccion proporcional de `CUDA allocated` ni speedup real de kernel.

## Modelo y entorno

Modelo usado:

```text
Qwen2.5-0.5B-Instruct
```

Ruta local:

```text
/home/neo/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

Entorno recomendado:

```bash
/home/neo/miniconda3/bin/python
```

GPU detectada fuera del sandbox:

```text
NVIDIA GeForce RTX 4060 Ti, 16 GB
```

## Que ya esta implementado

### 1. QPP con compresion real de parametros

Archivo:

```text
qpp_true_compression.py
```

Este script:

- divide matrices `Linear[out, in]` en bloques de filas;
- guarda una sola permutacion de columnas por bloque;
- ajusta `K` anchors por fila, default `K=16`;
- opcionalmente guarda outliers top-k;
- calibra anchors con activaciones;
- reconstruye pesos densos para medir PPL;
- acepta o rechaza modulos con un gate de delta PPL;
- genera CSV, JSON, Markdown y PNG.

Este script sirve para medir calidad y compresion teorica/real del formato, pero no mide ejecucion comprimida pura porque reconstruye a denso.

### 2. Runtime con modulos comprimidos reales

Archivo:

```text
qpp_runtime_eval.py
```

Este script reemplaza `nn.Linear` por:

```text
QPPCompressedLinear
```

El modulo comprimido guarda:

- `anchors`;
- `orders_i16`;
- `basis`;
- bias si existe;
- outlier indices/values si se habilitan.

No conserva el peso denso como parametro persistente.

En `forward()`, por ahora reconstruye temporalmente el peso denso y llama a `F.linear`.

Eso significa:

```text
Si: ejecucion con parametros persistentes comprimidos.
No: kernel eficiente todavia.
```

## Resultados principales

### Compresion real reconstruida densa

Artefactos:

```text
outputs/qpp_true_compression_rollup/
```

Tabla:

| Run | Target | Calibrado | Aceptados | BF16 PPL | QPP PPL | Delta PPL | Compresion real |
|---|---|---:|---:|---:|---:|---:|---:|
| attention2 smoke | attention | no | 1/2 | 5.2979 | 5.5015 | +0.2036 | 38.96x |
| attention2 calib | attention | si | 2/2 | 5.2979 | 5.3715 | +0.0736 | 38.96x |
| attention8 calib | attention | si | 7/8 | 4.5596 | 5.1731 | +0.6135 | 38.96x |
| attention8 calib topk4 | attention | si | 7/8 | 4.5596 | 5.1156 | +0.5560 | 28.90x |
| mlp6 calib | mlp | si | 2/6 | 4.5596 | 4.8339 | +0.2743 | 44.86x |

Lectura:

- La calibracion por activaciones fue clave.
- Attention es mas estable que MLP para compresion parcial.
- MLP tiene modulos que comprimen muy bien y otros que rompen PPL.
- `attention8` quedo cerca del gate `Delta PPL <= +0.5`, pero no lo pasa aun.

### Runtime con parametros comprimidos

Artefactos:

```text
outputs/qpp_runtime_rollup/
```

Tabla:

| Run | Aceptados | Tokens | BF16 PPL | QPP Runtime PPL | Delta PPL | BF16 eval | QPP eval | Compresion runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| runtime attention2 calib | 2/2 | 4096 | 5.2979 | 5.3730 | +0.0751 | 0.5470s | 0.3329s | 37.33x |
| runtime attention8 calib | 7/8 | 8192 | 4.6657 | 5.4671 | +0.8014 | 0.9090s | 0.7305s | 37.77x |
| runtime attention8 anchors32 long | 7/8 | 32768 | 4.5423 | 4.7631 | +0.2208 | 3.1352s | 3.0211s | 22.56x |
| runtime qkv24 anchors32 gate0.5 | 11/24 | 8192 | 4.6657 | 5.1571 | +0.4914 | 0.9074s | 0.7457s | 22.40x |

Lectura:

- El modelo si ejecuta con modulos QPP comprimidos reales.
- La compresion persistente de esos modulos es aproximadamente `37x`.
- La mejor configuracion actual de calidad es `anchors=32,row_block=128`: reduce compresion a `22.56x`, pero baja Delta PPL a `+0.2208`.
- La mejor configuracion actual orientada a VRAM es `q/k/v`, saltando `o_proj`, con gate global `Delta PPL <= 0.5`: acepta 11/24 modulos, reduce bytes persistentes del modelo en `8.995 MB`, y mantiene Delta PPL `+0.4914`.
- La medicion de tiempo no debe venderse como speedup real, porque:
  - la evaluacion es corta;
  - hay efectos de warmup/cache;
  - el forward reconstruye pesos temporales;
  - no existe todavia kernel QPP directo.

## Memoria: punto importante

Hay que distinguir dos cosas.

### Memoria persistente de los modulos comprimidos

Aqui si baja fuerte.

Para `runtime attention2 calib`:

```text
BF16 denso de modulos aceptados: 1.835 MB
QPP runtime buffers:            0.049 MB
Compresion real:                37.33x
```

Para `runtime attention8 calib`:

```text
BF16 denso de modulos aceptados: 5.734 MB
QPP runtime buffers:            0.152 MB
Compresion real:                37.77x
```

Para `runtime attention8 anchors32 long`:

```text
BF16 denso de modulos aceptados: 5.734 MB
QPP runtime buffers:            0.254 MB
Compresion real:                22.56x
Delta PPL:                      +0.2208
```

### VRAM total del proceso

Aqui todavia no baja claramente.

Medicion observada:

```text
CUDA allocated after BF16 load:       ~988 MB
CUDA allocated after QPP runtime eval: ~994-996 MB
```

Medicion nueva con `qkv24 anchors32 gate0.5` sin basis persistente:

```text
Persistent model bytes after BF16 load: 988.066 MB
Persistent model bytes after QPP:       979.071 MB
Reduccion persistente:                   -8.995 MB

CUDA allocated after BF16 load:          988.098 MB
CUDA allocated after QPP eval:           988.674 MB
```

Interpretacion:

- Los bytes persistentes del modelo si bajan.
- `CUDA allocated` no baja igual porque incluye cache, temporales y reconstruccion durante forward.
- La reduccion total visible seguira limitada hasta comprimir mas capas y eliminar reconstruccion densa temporal.

Razones:

- Solo comprimimos pocos modulos.
- El resto de Qwen sigue en BF16.
- PyTorch reserva/cacha memoria.
- `QPPCompressedLinear.forward()` reconstruye un peso temporal denso.
- Aun no existe kernel directo sobre anchors + shared order.

Conclusion honesta:

```text
Compresion real de parametros: si.
Menor VRAM total end-to-end: aun no demostrado.
Speedup real: aun no demostrado.
```

## Estado tecnico actual

Estamos en este punto:

```text
QPP ya no es solo una curva bonita.
QPP ya comprime parametros reales de algunos modulos Qwen.
QPP ya puede ejecutar el modelo con esos modulos comprimidos.
QPP todavia no tiene kernel eficiente ni compresion full-model estable.
```

Lo que ya es positivo:

- `attention2 calib`: Delta PPL `+0.0751`, compresion runtime `37.33x`.
- `attention8 calib`: 7 de 8 modulos aceptados, compresion runtime `37.77x`.
- `attention8 anchors32 long`: 7 de 8 modulos aceptados, Delta PPL `+0.2208`, compresion runtime `22.56x`.
- `qkv24 anchors32 gate0.5`: 11 de 24 modulos aceptados, Delta PPL `+0.4914`, compresion runtime `22.40x`, bytes persistentes `-8.995 MB`.
- `mlp6 calib`: 2 de 6 modulos aceptados, compresion real `44.86x`.

## Barrido por partes del modelo

Se probo cada familia con:

```text
anchors=32
total_delta_gate=0.5
eval_tokens=8192
calib_tokens=8192
```

Artefactos:

```text
outputs/qpp_partscan_rollup/
```

Resultados independientes por familia:

| Parte | Aceptados | Delta PPL | BF16 aceptado | QPP runtime | Ahorro persistente |
|---|---:|---:|---:|---:|---:|
| attn_q | 11/24 | +0.4949 | 17.662 MB | 0.788 MB | 16.893 MB |
| attn_k | 7/24 | +0.4944 | 1.606 MB | 0.072 MB | 1.536 MB |
| attn_v | 6/24 | +0.4909 | 1.376 MB | 0.061 MB | 1.316 MB |
| attn_o | 9/24 | +0.4420 | 14.451 MB | 0.629 MB | 13.822 MB |
| mlp_gate | 3/24 | +0.4421 | 26.149 MB | 1.138 MB | 25.011 MB |
| mlp_up | 2/24 | +0.4770 | 17.433 MB | 0.759 MB | 16.674 MB |
| mlp_down | 2/24 | +0.4920 | 17.433 MB | 0.387 MB | 17.046 MB |
| qkv_combo | 11/24 | +0.4914 | 9.404 MB | 0.420 MB | 8.995 MB |

Nota importante:

```text
Estos son escaneos independientes. No se pueden sumar directamente porque cada uno consume casi todo el presupuesto de Delta PPL 0.5.
```

Lectura:

- `attn_q` es la mejor parte de attention por ahorro individual.
- `attn_k` y `attn_v` pesan muy poco; aunque compriman bien, aportan poco a VRAM.
- `attn_o` puede ahorrar bastante, pero capa 0 es muy sensible.
- MLP es donde esta el dinero en memoria, pero solo pocas capas pasan el gate con este metodo.
- Bajo el metodo actual, el ahorro controlado por calidad esta en decenas de MB, no en cientos de MB.

Implicacion para bajar el modelo al 50%:

```text
Modelo BF16: ~988 MB
Objetivo 50%: ~494 MB
Ahorro necesario: ~494 MB

Mejor ahorro controlado actual por run: ~25 MB en una familia MLP independiente,
o ~9 MB en un run combinado q/k/v con gate global.
```

Conclusion:

```text
Con QPP shared-order actual no estamos cerca del 50%.
Para llegar al 50% hay que resolver MLP de forma mucho mas agresiva,
probablemente con residual/LoRA, INT residual, o un esquema hibrido.
```

Lo que falta:

- Ya se bajo `attention8` a Delta PPL `+0.2208` con `anchors=32`.
- `row_block=64` no mejoro contra `row_block=128`.
- `outlier_topk=4` no mejoro contra `anchors=32` sin outliers.
- `anchors=64` no rescato `layer0.self_attn.o_proj`; ese modulo sigue siendo sensible y conviene dejarlo BF16 por ahora.
- Implementar forward sin reconstruir peso denso.
- Medir en un corpus mas serio y mas largo.

## Archivos importantes

Scripts:

```text
qpp_true_compression.py
qpp_runtime_eval.py
qpp_qwen_quant_eval.py
qpp_mnist_analysis.py
test_qpp_true_compression.py
```

Reportes:

```text
outputs/qpp_true_compression_rollup/REPORT.md
outputs/qpp_runtime_rollup/REPORT.md
outputs/qpp_runtime_tuning_rollup/REPORT.md
outputs/qpp_vram_progress/REPORT.md
outputs/qpp_partscan_rollup/REPORT.md
outputs/qpp_qwen_summary/REPORT.md
outputs/qpp_mnist_digits_analysis/REPORT.md
outputs/conda_env_probe/REPORT.md
```

Imagenes:

```text
outputs/qpp_true_compression_rollup/qpp_true_compression_rollup.png
outputs/qpp_runtime_rollup/qpp_runtime_rollup.png
outputs/qpp_runtime_tuning_rollup/qpp_runtime_tuning_rollup.png
outputs/qpp_vram_progress/qpp_vram_progress.png
outputs/qpp_partscan_rollup/qpp_partscan_rollup.png
outputs/qpp_qwen_summary/qpp_qwen_runs_summary.png
outputs/qpp_mnist_digits_analysis/decision_table_precision_time_compression.png
outputs/conda_env_probe/conda_env_probe.png
```

## Comandos reproducibles

Smoke runtime con 2 modulos attention:

```bash
/home/neo/miniconda3/bin/python qpp_runtime_eval.py \
  --target attention \
  --max-modules 2 \
  --calibrate \
  --calib-tokens 4096 \
  --calib-rows 1024 \
  --eval-tokens 4096 \
  --text-repeat 8 \
  --generate \
  --outdir outputs/qpp_runtime_attention2_calib_clean
```

Runtime con 8 modulos attention:

```bash
/home/neo/miniconda3/bin/python qpp_runtime_eval.py \
  --target attention \
  --max-modules 8 \
  --calibrate \
  --calib-tokens 8192 \
  --calib-rows 1024 \
  --eval-tokens 8192 \
  --text-repeat 12 \
  --outdir outputs/qpp_runtime_attention8_calib
```

Mejor runtime actual con 8 modulos attention:

```bash
/home/neo/miniconda3/bin/python qpp_runtime_eval.py \
  --target attention \
  --max-modules 8 \
  --anchors 32 \
  --row-block 128 \
  --calibrate \
  --calib-tokens 32768 \
  --calib-rows 2048 \
  --eval-tokens 32768 \
  --text-repeat 40 \
  --generate \
  --outdir outputs/qpp_runtime_attention8_a32_calib_long
```

Mejor run actual orientado a VRAM con gate global:

```bash
/home/neo/miniconda3/bin/python qpp_runtime_eval.py \
  --target attention \
  --skip-substrings o_proj \
  --max-modules 24 \
  --anchors 32 \
  --row-block 128 \
  --calibrate \
  --calib-tokens 8192 \
  --calib-rows 1024 \
  --eval-tokens 8192 \
  --text-repeat 16 \
  --total-delta-gate 0.5 \
  --outdir outputs/qpp_runtime_attention_qkv24_a32_totalgate05_nobasis
```

Compresion reconstruida densa con 8 modulos attention:

```bash
/home/neo/miniconda3/bin/python qpp_true_compression.py \
  --target attention \
  --max-modules 8 \
  --eval-tokens 32768 \
  --calibrate \
  --calib-tokens 32768 \
  --calib-rows 2048 \
  --text-repeat 32 \
  --outdir outputs/qpp_true_attention8_calib
```

Tests ligeros:

```bash
python -m py_compile qpp_runtime_eval.py qpp_true_compression.py test_qpp_true_compression.py

python - <<'PY'
import test_qpp_true_compression as t
for name in sorted(n for n in dir(t) if n.startswith('test_')):
    getattr(t, name)()
    print(f'{name}: PASS')
PY
```

## Proximo paso recomendado

## Sweep nuevo de enfoques QPP

Se agrego soporte de residual low-rank contado dentro de `qpp_runtime_eval.py` mediante `--residual-rank`. El modulo comprimido ahora puede almacenar:

- anchors QPP;
- ordenes compartidos;
- outliers top-k opcionales;
- residual low-rank FP16 opcional;
- bias si existe.

El residual no es gratis: sus factores se cuentan en `runtime_buffer_bytes`, `theoretical_qpp_bytes`, CSV, JSON y PNG.

Artefactos:

```text
outputs/qpp_approach_sweep/REPORT.md
outputs/qpp_approach_sweep/qpp_approach_sweep_summary.csv
outputs/qpp_approach_sweep/qpp_approach_sweep_summary.png
```

Condiciones del sweep:

- 8192 tokens de evaluacion;
- 8192 tokens de calibracion cuando aplica;
- `anchors=32`;
- `total_delta_gate=0.5`;
- ejecucion real con modulos comprimidos en memoria;
- forward reconstruyendo temporalmente pesos densos.

Resultados principales:

| Grupo | Enfoque | Aceptados | Delta PPL | Delta modelo persistente | Compresion runtime subset |
|---|---:|---:|---:|---:|---:|
| attention | QPP calib + rank8 | 8/8 | +0.1847 | -6.840 MB | 14.55x |
| attention | QPP calib + topk8 | 6/8 | +0.2390 | -3.875 MB | 16.00x |
| attention | QPP calib | 7/8 | +0.2671 | -5.485 MB | 22.56x |
| attention | QPP no calib | 2/8 | +0.3735 | -3.071 MB | 22.40x |
| mlp | QPP calib + rank16 | 2/6 | +0.2288 | -16.491 MB | 18.51x |
| mlp | QPP calib | 2/6 | +0.2420 | -16.860 MB | 30.43x |
| mlp | QPP calib + rank8 | 3/6 | +0.2578 | -24.920 MB | 21.28x |
| mlp | QPP calib + topk16 | 3/6 | +0.2744 | -24.517 MB | 16.02x |

Lectura:

- La calibracion por activaciones es obligatoria; QPP sin calibracion queda claramente peor.
- En attention, `QPP calib + rank8` fue el mejor equilibrio de calidad y cobertura: acepta 8/8 modulos.
- En MLP, `QPP calib + rank8` fue el mejor ahorro del sweep: acepta 3/6 modulos y baja casi 25 MB persistentes en el subset.
- `rank16` no domino a `rank8`: mejora algo PPL en MLP, pero acepta menos modulos y ahorra menos memoria.
- Las primeras capas MLP siguen siendo el bloqueo fuerte: `layer0.mlp.gate_proj`, `up_proj` y `down_proj` rompen PPL incluso con top-k o residual low-rank pequeno.

Conclusion nueva: el enfoque hibrido QPP + residual low-rank si destaca frente a QPP puro para coverage, especialmente en attention y algunos MLP. Pero todavia no alcanza el objetivo de reducir el modelo completo a 50% de VRAM. Para eso hay que resolver MLP temprano y embeddings, o aceptar una tecnica hibrida mas pesada/entrenada.

## Experimento fuerte: residual low-rank entrenable

Se extendio `qpp_runtime_eval.py` con:

```text
--train-residual-steps
--train-residual-lr
--train-residual-batch
```

El entrenamiento congela QPP base y ajusta solo los factores low-rank `A/B` del residual usando activaciones reales. La perdida local es MSE sobre la salida residual del lineal:

```text
X @ (W_original - W_qpp).T ~= (X @ B.T) @ A.T
```

Esto sigue siendo una calibracion local por lineal, no fine-tuning CE/KL del modelo completo.

Artefactos:

```text
outputs/qpp_trainres_summary/REPORT.md
outputs/qpp_trainres_summary/qpp_trainres_summary.csv
outputs/qpp_trainres_summary/qpp_trainres_summary.png
```

Resultados:

| Grupo | Enfoque | Aceptados | Delta PPL | BF16 aceptado | QPP runtime | Compresion subset | Delta persistente |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp6 | rank8 entrenado | 3/6 | +0.2224 | 26.149 MB | 1.229 MB | 21.28x | -24.920 MB |
| mlp6 | rank16 entrenado | 3/6 | +0.1985 | 26.149 MB | 1.505 MB | 17.37x | -24.644 MB |
| mlp6 | rank32 entrenado | 3/6 | +0.1324 | 26.149 MB | 2.058 MB | 12.70x | -24.091 MB |
| layer0 MLP | rank64 entrenado | 0/3 | +0.0000 | 0.000 MB | 0.000 MB | 0.00x | 0.000 MB |
| layer0 MLP | rank128 entrenado | 1/3 | +0.2297 | 8.716 MB | 1.854 MB | 4.70x | -6.862 MB |
| layer0 MLP | rank256 entrenado | 3/3 | +0.3263 | 26.149 MB | 9.800 MB | 2.67x | -16.349 MB |
| mlp6 | rank256 entrenado | 5/6 | +0.1937 | 43.581 MB | 16.457 MB | 2.65x | -27.125 MB |

Lectura importante:

- Entrenar el residual si mejora frente al residual SVD estatico.
- Rank32 mejora calidad en MLP6 sin cambiar coverage: 3/6 modulos aceptados, Delta PPL baja a `+0.1324`.
- Rank256 es el primer setting que rescata los tres `layer0.mlp.*` cuando se prueban juntos, pero la compresion baja a `~2.67x`.
- En MLP6 rank256 se aceptan 5/6, con Delta PPL `+0.1937`; aun asi, `layer0.mlp.down_proj` se rechaza cuando ya estan aceptados `gate/up`, porque el error acumulado pasa el gate.
- Esto confirma que el bloqueo no es solo "QPP no tiene capacidad"; con residual grande si funciona. El problema es el tradeoff calidad/bytes y la interaccion no lineal del bloque MLP.

Conclusion actualizada:

Para bajar mucho VRAM con buena PPL, la ruta ya no es QPP puro. La ruta plausible es hibrida/adaptativa:

```text
attention: QPP calib + rank8
MLP tolerante: QPP calib + rank8/rank32
MLP temprano sensible: QPP calib + rank128/rank256 o entrenamiento de bloque
embeddings/lm_head: INT8/INT4 u otro esquema especializado
```

El siguiente salto real es seleccionar rango por modulo automaticamente y/o entrenar el bloque MLP completo con perdida de lenguaje, porque el ajuste local por lineal ya demostro que recupera salida local pero no siempre controla PPL acumulada.

## Full run sobre todos los lineales target

Se ejecuto un pase completo sobre todos los lineales target del transformer (`attention + MLP`) con:

```text
target=all
max_modules=0
anchors=32
residual_rank=256
train_residual_steps=200
total_delta_gate=0.5
eval_tokens=8192
calib_tokens=8192
generate=true
```

Artefactos:

```text
outputs/qpp_full_all_rank256_s200_gate05/REPORT.md
outputs/qpp_full_all_rank256_s200_gate05/qpp_runtime_results.json
outputs/qpp_full_all_rank256_s200_gate05/qpp_runtime_decisions.csv
outputs/qpp_full_all_rank256_s200_gate05/qpp_runtime_decisions.png
outputs/qpp_full_summary/REPORT.md
outputs/qpp_full_summary/qpp_full_summary.csv
outputs/qpp_full_summary/qpp_full_summary.png
```

Resultado total contra el modelo BF16 original:

| Run | Modulos aceptados | BF16 PPL | QPP PPL | Delta PPL | Modelo BF16 | Modelo QPP | Ahorro MB | Ahorro total | Modelo restante | Compresion subset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full all rank256 trained | 112/168 | 4.6657 | 5.1353 | +0.4696 | 988.066 MB | 814.566 MB | 173.499 MB | 17.56% | 82.44% | 2.22x |

Lectura:

- Este es el mejor resultado completo medido hasta ahora.
- Ya no es un experimento parcial: se intento todo `attention + MLP` y se acepto lo que paso el gate global de calidad.
- Se obtuvo una reduccion persistente real de `173.499 MB`, equivalente a `17.56%` del modelo BF16 original.
- El modelo comprimido persistente queda en `814.566 MB`, o `82.44%` del original.
- La PPL queda dentro del gate: Delta PPL `+0.4696`.
- Generacion real ejecuto, pero el texto generado fue pobre (`"Explain quantization in one short paragraph. TP"`), asi que la validacion de generacion todavia no es suficiente para afirmar calidad conversacional.
- La velocidad empeoro: slowdown `1.855x` vs BF16, porque el forward reconstruye pesos densos temporalmente.

Punto tecnico importante:

Este full run usa `rank256` para todos los modulos. Eso rescata calidad, pero es ineficiente para matrices pequeñas como `k_proj/v_proj`, donde se vio compresion menor a `1x`. Por eso el siguiente paso no es subir mas rango globalmente, sino usar seleccion adaptativa:

```text
attention k/v: rank0/rank8 o QPP puro
attention q/o: rank8/rank32
MLP tolerante: rank8/rank32
MLP sensible: rank128/rank256
modulos donde rank256 no ahorra: dejar BF16
embeddings/lm_head: cuantizacion separada INT8/INT4
```

Para llegar a 50% todavia falta atacar embeddings/lm_head y optimizar la politica por rango. Este full run demuestra `17.56%` real, no 50%.

## Full run Qwen2-1.5B

Se probo tambien el modelo local:

```text
/home/neo/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a
```

Configuracion comparable al full run de 0.5B:

```text
target=all
max_modules=0
anchors=32
residual_rank=256
train_residual_steps=200
total_delta_gate=0.5
eval_tokens=8192
calib_tokens=8192
generate=true
```

Artefactos:

```text
outputs/qpp_qwen2_1p5b_full_all_rank256_s200_gate05/REPORT.md
outputs/qpp_qwen2_1p5b_full_all_rank256_s200_gate05/qpp_runtime_results.json
outputs/qpp_qwen2_1p5b_full_all_rank256_s200_gate05/qpp_runtime_decisions.csv
outputs/qpp_qwen2_1p5b_full_all_rank256_s200_gate05/qpp_runtime_decisions.png
outputs/qpp_model_size_comparison/REPORT.md
outputs/qpp_model_size_comparison/qpp_model_size_comparison.csv
outputs/qpp_model_size_comparison/qpp_model_size_comparison.png
```

Comparativa directa:

| Modelo | Modulos aceptados | BF16 PPL | QPP PPL | Delta PPL | Modelo BF16 | Modelo QPP | Ahorro MB | Ahorro total | Modelo restante | Compresion subset | Slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 112/168 | 4.6657 | 5.1353 | +0.4696 | 988.066 MB | 814.566 MB | 173.499 MB | 17.56% | 82.44% | 2.22x | 1.86x |
| Qwen2-1.5B | 140/196 | 4.0494 | 4.4944 | +0.4449 | 3087.429 MB | 2100.910 MB | 986.520 MB | 31.95% | 68.05% | 3.64x | 2.02x |

Lectura:

- El Qwen2-1.5B comprime bastante mejor que el 0.5B bajo la misma politica global.
- Ahorro total real persistente: `986.520 MB`, equivalente a `31.95%` del modelo BF16 original.
- El modelo restante queda en `2100.910 MB`, o `68.05%` del original.
- PPL se mantiene dentro del gate global `+0.5`: Delta PPL `+0.4449`.
- Se aceptaron `140/196` lineales target.
- La generacion real ejecuto, pero el texto fue malo/repetitivo (`"100000..."`), asi que PPL local no basta como validacion conversacional.
- La velocidad sigue peor que BF16 (`2.02x` mas lento) por reconstruccion densa temporal.

Conclusion nueva:

La hipotesis de que un modelo mas grande puede tolerar mejor QPP+hibrido se confirma parcialmente: el 1.5B logro `31.95%` de ahorro total contra `17.56%` en 0.5B. Todavia no llega a 50%, pero ya esta mucho mas cerca. El siguiente salto debe ser:

```text
1. politica adaptativa de rank por modulo para no usar rank256 donde no ahorra;
2. evitar comprimir k/v con rank256 si runtime_comp < 1x;
3. cuantizar embeddings/lm_head por separado;
4. validar con prompts/generacion mas estrictos, no solo PPL local;
5. implementar forward directo para eliminar reconstruccion densa.
```

## Pruebas con modelos mas grandes/no-Qwen

### Qwen3-4B dense

Se descargo y probo el modelo recomendado:

```text
Qwen/Qwen3-4B
/home/neo/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

Probe:

- Carga BF16 completa: si.
- Peso persistente BF16: `8044.937 MB`.
- Lineales target: `252`.
- PPL corto 2048 tokens: `8.0067`, sin OOM.

Configuracion full QPP:

```text
target=all
max_modules=0
anchors=32
residual_rank=256
train_residual_steps=200
total_delta_gate=0.5
eval_tokens=8192
calib_tokens=8192
generate=true
```

Artefactos:

```text
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05/REPORT.md
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05/qpp_runtime_results.json
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05/qpp_runtime_decisions.csv
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05/qpp_runtime_decisions.png
outputs/qpp_model_size_comparison/REPORT.md
outputs/qpp_model_size_comparison/qpp_model_size_comparison.csv
outputs/qpp_model_size_comparison/qpp_model_size_comparison.png
```

Resultado:

| Modelo | Modulos aceptados | BF16 PPL | QPP PPL | Delta PPL | Modelo BF16 | Modelo QPP | Ahorro MB | Ahorro total | Modelo restante | Compresion subset | Slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 103/252 | 3.9187 | 4.3939 | +0.4752 | 8044.937 MB | 5911.371 MB | 2133.565 MB | 26.52% | 73.48% | 5.81x | 2.23x |

Desglose de ahorro por tipo aceptado:

| Tipo | Aceptados | BF16 aceptado | QPP | Ahorro | Compresion |
|---|---:|---:|---:|---:|---:|
| `mlp.up_proj` | 14 | 697.30 MB | 102.24 MB | 595.06 MB | 6.82x |
| `mlp.gate_proj` | 14 | 697.30 MB | 102.24 MB | 595.06 MB | 6.82x |
| `attn.q_proj` | 22 | 461.37 MB | 84.34 MB | 377.03 MB | 5.47x |
| `attn.o_proj` | 14 | 293.60 MB | 52.30 MB | 241.30 MB | 5.61x |
| `mlp.down_proj` | 5 | 249.04 MB | 36.17 MB | 212.87 MB | 6.89x |
| `attn.v_proj` | 21 | 110.10 MB | 40.77 MB | 69.33 MB | 2.70x |
| `attn.k_proj` | 13 | 68.16 MB | 25.24 MB | 42.92 MB | 2.70x |

Lectura:

- Qwen3-4B es el mejor en ahorro absoluto: `2133.565 MB`.
- En porcentaje total no supera a Qwen2-1.5B: `26.52%` vs `31.95%`.
- Los modulos aceptados comprimen muy bien: `5.81x` en el subset aceptado.
- Qwen3-4B tiene `k/v` que si comprimen con rank256 (`~2.70x`), a diferencia de Qwen2-1.5B donde `k/v` quedaban por debajo de `1x`.
- El problema fue calidad/cobertura: solo acepto `103/252` modulos bajo gate `+0.5`.
- Generacion real salio repetitiva: `"The model is the model..."`, por lo que PPL local aun no valida calidad conversacional.

### Qwen3-4B dense con forward directo

Se repitio el full QPP sobre Qwen3-4B usando:

```text
forward_mode=direct
target=all
max_modules=0
anchors=32
residual_rank=256
train_residual_steps=200
total_delta_gate=0.5
eval_tokens=8192
calib_tokens=8192
generate=true
save_compressed_artifact=true
```

Artefactos:

```text
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05_direct/REPORT.md
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05_direct/qpp_runtime_results.json
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05_direct/qpp_runtime_decisions.csv
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05_direct/qpp_runtime_decisions.png
outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05_direct/qpp_compressed_artifact.pt
outputs/qpp_qwen3_4b_bf16_generation_baseline/REPORT.md
outputs/qpp_qwen3_direct_reference_summary/REPORT.md
outputs/qpp_qwen3_direct_reference_summary/qwen3_qpp_direct_reference.csv
outputs/qpp_qwen3_direct_reference_summary/qwen3_qpp_direct_reference.png
```

Resultado comparativo:

| Modo | Aceptados | PPL | Delta PPL | Modelo MB | Ahorro | Eval s | Eval slowdown | Gen 64 tok s | tok/s | Gen slowdown | Artefacto | Nota |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF16 | 0/252 | 3.9187 | 0.0000 | 8044.9 | 0.00% | 4.181 | 1.00x | 2.601 | 24.61 | 1.00x | 0.0 MB | baseline |
| QPP reconstruct | 103/252 | 4.3939 | +0.4752 | 5911.4 | 26.52% | 9.320 | 2.23x | 24.918 | 2.57 | 9.58x | 0.0 MB | repetitivo |
| QPP direct | 98/252 | 4.4106 | +0.4919 | 6013.7 | 25.25% | 6.559 | 1.56x | 9.720 | 6.58 | 3.74x | 443.7 MB | repetitivo |

Lectura:

- Si, QPP direct usa menos memoria persistente que BF16: `8044.9 MB -> 6013.7 MB`, ahorro `2031.2 MB` (`25.25%`).
- Tambien baja CUDA allocated tras evaluacion frente a BF16 baseline de generacion: alrededor de `8053 MB -> 6065 MB`; CUDA reserved queda alto por cache de PyTorch y no debe leerse como memoria real del modelo.
- Direct mejora claramente la velocidad contra QPP reconstruct: evaluacion `9.320s -> 6.559s`; generacion `24.918s -> 9.720s`.
- Direct todavia no alcanza BF16: evaluacion `1.56x` mas lenta; generacion autoregresiva `3.74x` mas lenta.
- La calidad de generacion sigue mala/repetitiva (`"The first permutation..."`), por lo que este artefacto queda como referencia tecnica de compresion/runtime, no como modelo listo para usar.
- `qpp_compressed_artifact.pt` no es un checkpoint HuggingFace standalone. Guarda reemplazos QPP comprimidos y referencia el modelo BF16 local base.

### Qwen2-7B

Modelo local:

```text
/home/neo/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e
```

Resultado del probe:

- Carga BF16 completa: si.
- Peso persistente BF16: `15231.234 MB`.
- CUDA allocated tras carga: `15232.283 MB`.
- Lineales target: `196`.
- PPL corto: no pudo ejecutarse por OOM.

Error observado:

```text
CUDA out of memory. Tried to allocate 298 MiB.
GPU total 15.57 GiB, free 152 MiB.
```

Conclusion: Qwen2-7B esta completo localmente, pero BF16 deja la GPU casi llena. Con el pipeline actual no se puede evaluar ni cuantizar full porque el forward necesita memoria adicional para logits/activaciones y QPP reconstruye pesos densos temporalmente. Para probar 7B haria falta una de estas rutas:

```text
1. cargar con device_map CPU/GPU offload;
2. baseline en 8-bit/4-bit y adaptar QPP por bloques;
3. cuantizar offline por shards sin cargar todo en GPU;
4. implementar forward QPP directo sin reconstruccion densa;
5. usar GPU con mas VRAM.
```

### Phi-2

Modelo local:

```text
/home/neo/.cache/huggingface/hub/models--microsoft--phi-2/snapshots/810d367871c1d460086d9f82db8696f2e0a0fcd0
```

Configuracion:

```text
target=all
max_modules=0
anchors=32
residual_rank=256
train_residual_steps=200
total_delta_gate=0.5
eval_tokens=8192
calib_tokens=8192
generate=true
```

Artefactos:

```text
outputs/qpp_phi2_full_all_rank256_s200_gate05/REPORT.md
outputs/qpp_phi2_full_all_rank256_s200_gate05/qpp_runtime_results.json
outputs/qpp_phi2_full_all_rank256_s200_gate05/qpp_runtime_decisions.csv
outputs/qpp_phi2_full_all_rank256_s200_gate05/qpp_runtime_decisions.png
outputs/qpp_model_size_comparison/REPORT.md
outputs/qpp_model_size_comparison/qpp_model_size_comparison.csv
outputs/qpp_model_size_comparison/qpp_model_size_comparison.png
```

Comparativa actual:

| Modelo | Modulos aceptados | BF16 PPL | QPP PPL | Delta PPL | Modelo BF16 | Modelo QPP | Ahorro MB | Ahorro total | Modelo restante | Compresion subset | Slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 112/168 | 4.6657 | 5.1353 | +0.4696 | 988.066 MB | 814.566 MB | 173.499 MB | 17.56% | 82.44% | 2.22x | 1.86x |
| Qwen2-1.5B | 140/196 | 4.0494 | 4.4944 | +0.4449 | 3087.429 MB | 2100.910 MB | 986.520 MB | 31.95% | 68.05% | 3.64x | 2.02x |
| Phi-2 | 80/192 | 2.9680 | 3.4653 | +0.4973 | 5559.368 MB | 4464.097 MB | 1095.270 MB | 19.70% | 80.30% | 5.08x | 1.81x |

Lectura:

- Phi-2 comprime mucho mejor los modulos aceptados (`5.08x` subset), pero acepta menos modulos (`80/192`) bajo el gate `+0.5`.
- Qwen2-1.5B sigue siendo el mejor resultado total hasta ahora: `31.95%` de ahorro del modelo completo.
- Qwen2-7B es el siguiente candidato natural, pero no cabe funcionalmente en BF16 con este pipeline en 16 GB VRAM.

El siguiente paso no debe ser comprimir mas capas a ciegas.

La ruta mas razonable:

1. Tomar `anchors=32,row_block=128` como baseline nuevo.
2. Extender seleccion `q/k/v` a mas capas con gate global de calidad.
3. Mantener `o_proj` fuera por ahora salvo que tenga tratamiento especial.
4. Para perseguir 50%, cambiar foco a MLP con metodo hibrido:
   - QPP base + residual bajo rango;
   - QPP base + residual INT4/INT8 sparse;
   - calibracion por salida del bloque MLP completo, no solo lineal individual.
5. Implementar forward directo sin reconstruir `W`.
6. Medir VRAM total y velocidad solo despues de tener computo directo sobre anchors + orden compartido.

## Forward directo experimental

Se implemento `--forward-mode direct` en `qpp_runtime_eval.py`.

Que hace:

- evita materializar el peso denso `W` para la parte QPP cuando no hay outliers;
- computa por bloque como `x[:, order] @ basis @ anchors.T`;
- aplica el residual low-rank como `(x @ B.T) @ A.T`;
- sigue usando operaciones PyTorch normales, no un kernel CUDA/Triton fusionado.

Validacion sintetica:

```text
max_abs_error ~= 0.0078 BF16
mean_abs_error ~= 0.0017 BF16
```

Smoke en `Qwen2.5-0.5B-Instruct`, `anchors=32`, `residual_rank=8`, `eval_tokens=4096`:

| Grupo | Modo | Aceptados | QPP PPL | Delta PPL | QPP s | Speedup vs reconstruct | MB QPP runtime | Comp runtime | Delta persistente |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attention8 | reconstruct | 7/8 | 8.5377 | +0.2399 | 0.3965 | 1.000x | 0.406 MB | 14.13x | -5.333 MB |
| attention8 | direct | 7/8 | 8.5388 | +0.2410 | 0.3927 | 1.010x | 0.807 MB | 7.10x | -4.932 MB |
| mlp6 | reconstruct | 1/6 | 8.7754 | +0.4775 | 0.4063 | 1.000x | 0.472 MB | 18.48x | -8.245 MB |
| mlp6 | direct | 1/6 | 8.6910 | +0.3932 | 0.3962 | 1.026x | 0.529 MB | 16.48x | -8.187 MB |

Artefactos:

```text
outputs/qpp_direct_forward_summary/REPORT.md
outputs/qpp_direct_forward_summary/qpp_direct_forward_summary.csv
outputs/qpp_direct_forward_summary/qpp_direct_forward_summary.png
outputs/qpp_direct_smoke_attn8_reconstruct/
outputs/qpp_direct_smoke_attn8_direct/
outputs/qpp_direct_smoke_mlp6_reconstruct/
outputs/qpp_direct_smoke_mlp6_direct/
```

Lectura honesta:

- El forward directo funciona y mantiene PPL casi igual al modo de reconstruccion.
- El speedup medido es chico: `~1%` en attention8 y `~2.6%` en mlp6 en estos smokes cortos.
- La compresion runtime baja en attention porque esta version guarda `basis` por modulo; eso duplica parte del metadata.
- Esto prueba la ruta matematica, pero no es todavia el kernel que necesitamos para acelerar fuerte.

Para acelerar de verdad falta fusionar el computo:

1. kernel Triton/CUDA que haga gather por `shared_order`, interpolacion/anchors y acumulacion en una sola pasada;
2. no guardar `basis` por modulo, sino generarla o compartirla por shape/configuracion;
3. soportar batches/tokens grandes sin matrices temporales intermedias;
4. medir tokens/s en generacion autoregresiva, no solo PPL batch.

## Primer perfil competitivo contra BF16

Se optimizo el forward directo con dos cambios:

- cache de indices `orders_i32` para evitar convertir `orders_i16 -> int32/int64` en cada forward;
- fast-path vectorizado para batches pequenos de generacion, procesando todos los bloques de un modulo con einsum batched en vez de un loop Python por bloque.

Resultado: QPP full direct+vectorized ya supera BF16 en tok/s, pero la generacion queda rota/repetitiva. El perfil que si queda competitivo como referencia util es `attention-only`.

Tabla actual:

| Modo | Modulos | PPL | Delta PPL | Ahorro | Modelo MB | Eval s | tok/s | Speed vs BF16 | Calidad smoke |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF16 | 0 | 3.9187 | 0.0000 | 0.00% | 8044.9 | 4.208 | 24.61 | 1.00x | baseline |
| QPP full direct+vec | 98 | 4.4106 | +0.4919 | 24.74% | 6054.7 | 6.532 | 29.55 | 1.20x | mala/repetitiva |
| QPP attention-only | 66 | 3.9233 | +0.0046 | 8.25% | 7381.2 | 4.880 | 26.53 | 1.08x | coherente smoke |
| QPP attn+lateMLP | 84 | 3.9992 | +0.0805 | 17.52% | 6635.4 | 5.853 | 28.80 | 1.17x | repetitiva |
| QPP attn+MLP26/31 | 72 | 3.9371 | +0.0183 | 11.32% | 7134.2 | 5.430 | 26.76 | 1.09x | repetitiva |

Artefacto competitivo guardado:

```text
outputs/qpp_qwen3_4b_attention_only_competitive_artifact/qpp_compressed_artifact.pt
outputs/qpp_qwen3_4b_attention_only_competitive_artifact/REPORT.md
outputs/qpp_qwen3_4b_attention_only_competitive_artifact_bench/REPORT.md
outputs/qpp_speed_quality_summary/REPORT.md
outputs/qpp_speed_quality_summary/qpp_speed_quality_summary.csv
outputs/qpp_speed_quality_summary/qpp_speed_quality_summary.png
```

Validacion del artefacto attention-only separado:

```text
PPL: 3.9233
delta PPL vs BF16: +0.0046
modelo persistente: 8044.9 MB -> 7381.2 MB
ahorro: 663.7 MB, 8.25%
generacion: 26.31 tok/s vs 24.61 tok/s BF16
speed vs BF16: 1.07x
texto smoke: coherente
```

Decision:

- Ya existe un perfil QPP que usa menos VRAM y supera BF16 en generacion corta, manteniendo calidad smoke: `QPP attention-only`.
- El perfil full comprime mucho mas y es rapido, pero no es usable por repeticion.
- El siguiente problema ya no es solo runtime; es seleccion/calidad. Hay que incorporar un gate de repeticion/generacion y no aceptar MLP solo por PPL.

## Frase final

El proyecto ya demostro compresion verdadera parcial en Qwen y un primer perfil competitivo (`QPP attention-only`) que reduce VRAM y supera BF16 en generacion corta. El cuello actual ya no es solo runtime: el problema fuerte es ampliar ahorro hacia MLP sin provocar repeticion ni degradacion conversacional.
