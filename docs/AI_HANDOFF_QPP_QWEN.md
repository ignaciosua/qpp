# AI Handoff: QPP / BitTrit Quantization For Qwen

Fecha: 2026-06-28

Este documento es el punto de entrada recomendado para otra IA o investigador que necesite entender rapido el proyecto, reproducir los resultados principales y continuar el trabajo.

## Objetivo

Investigar si el enfoque QPP / BitTrit puede comprimir modelos tipo Qwen de forma real:

- menos bytes persistentes del modelo;
- menor uso de VRAM;
- PPL razonable;
- generacion no rota;
- velocidad competitiva o superior a BF16.

La idea inicial era que los perceptrones/pesos lineales podian representarse con curvas cuantiles compactas. El hallazgo central fue que la compresion solo es real si se evita guardar una permutacion por fila.

## Idea tecnica

El primer QPP usaba `argsort` por fila. Eso parecia comprimir mucho porque cada fila quedaba descrita por pocos parametros, pero al guardar una permutacion por fila la metadata destruia la compresion: cerca de `1x`.

El enfoque actual usa:

- `shared_order` por bloque de filas, no por fila;
- `anchors` por fila, normalmente `K=32`;
- calibracion por activaciones;
- residual low-rank opcional;
- seleccion greedy por modulo con gate de PPL;
- runtime real con `QPPCompressedLinear`;
- forward directo sin reconstruir `W` denso para la parte QPP;
- fast-path vectorizado para generacion corta.

## Estado actual

Ya existe un perfil competitivo contra BF16 en Qwen3-4B:

| Perfil | Modulos QPP | PPL | Delta PPL | Ahorro modelo | tok/s gen | Speed vs BF16 | Calidad smoke |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | 0 | 3.9187 | 0.0000 | 0.00% | 24.61 | 1.00x | baseline |
| QPP full direct+vec | 98 | 4.4106 | +0.4919 | 24.74% | 29.55 | 1.20x | mala/repetitiva |
| QPP attention-only | 66 | 3.9233 | +0.0046 | 8.25% | 26.53 | 1.08x | coherente smoke |
| QPP attn+lateMLP | 84 | 3.9992 | +0.0805 | 17.52% | 28.80 | 1.17x | repetitiva |
| QPP attn+MLP26/31 | 72 | 3.9371 | +0.0183 | 11.32% | 26.76 | 1.09x | repetitiva |

Conclusion actual:

- `QPP attention-only` es el primer resultado usable: reduce VRAM, supera BF16 en generacion corta y mantiene salida coherente en el smoke.
- `QPP full direct+vec` comprime mucho mas y es rapido, pero rompe calidad por repeticion.
- El cuello actual ya no es solo runtime. El problema principal es ampliar compresion hacia MLP sin provocar repeticion.

## Artefacto recomendado

Artefacto QPP competitivo:

```text
outputs/qpp_qwen3_4b_attention_only_competitive_artifact/qpp_compressed_artifact.pt
```

Este artefacto:

- contiene 66 modulos de atencion comprimidos;
- pesa aproximadamente `195 MB` en disco;
- referencia el modelo Qwen3-4B BF16 local;
- no es un checkpoint HuggingFace standalone;
- debe cargarse reemplazando modulos `nn.Linear` por `QPPCompressedLinear`.

Modelo base local:

```text
/home/neo/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

## Entorno

Python recomendado:

```bash
/home/neo/miniconda3/bin/python
```

GPU usada:

```text
NVIDIA GeForce RTX 4060 Ti, 16 GB
```

Notas:

- Para cargar modelos de HuggingFace cache y usar CUDA, ejecutar fuera del sandbox.
- Triton esta instalado (`3.6.0`), pero todavia no hay kernel Triton fusionado de produccion.

## Archivos principales

| Archivo | Funcion |
|---|---|
| `qpp_true_compression.py` | Evalua formato QPP con reconstruccion densa para PPL/calidad. |
| `qpp_runtime_eval.py` | Runtime real con `QPPCompressedLinear`, export de artefactos y forward directo. |
| `qpp_artifact_benchmark.py` | Carga un artefacto QPP sin recalibrar y mide PPL/generacion/VRAM. |
| `qpp_filter_artifact.py` | Filtra un artefacto QPP por subconjunto de modulos. |
| `qpp_speed_quality_summary.py` | Genera tabla/CSV/PNG de perfiles speed/calidad. |
| `qpp_bf16_generation_benchmark.py` | Baseline BF16 de generacion. |
| `ESTADO_ACTUAL_QPP_QWEN.md` | Bitacora completa del proyecto y resultados historicos. |
| `QPP_BitTrit_BGSP_documentacion_completa.md` | Documentacion amplia del enfoque original. |

## Como reproducir el perfil competitivo

Validar el artefacto `attention-only`:

```bash
/home/neo/miniconda3/bin/python qpp_artifact_benchmark.py \
  --artifact outputs/qpp_qwen3_4b_attention_only_competitive_artifact/qpp_compressed_artifact.pt \
  --outdir outputs/qpp_qwen3_4b_attention_only_competitive_artifact_bench \
  --device cuda \
  --eval-tokens 8192 \
  --ctx 1024 \
  --text-repeat 16 \
  --generate \
  --max-new-tokens 64 \
  --cache-indices \
  --direct-vectorize-max-tokens 16
```

Resultado esperado aproximado:

```text
PPL: 3.9233
Delta PPL vs BF16: +0.0046
Modelo persistente: 8044.9 MB -> 7381.2 MB
Ahorro: 663.7 MB, 8.25%
Generacion: ~26.3 tok/s
BF16 generacion: ~24.6 tok/s
```

Generar resumen comparativo:

```bash
python qpp_speed_quality_summary.py
```

Salidas:

```text
outputs/qpp_speed_quality_summary/REPORT.md
outputs/qpp_speed_quality_summary/qpp_speed_quality_summary.csv
outputs/qpp_speed_quality_summary/qpp_speed_quality_summary.png
```

## Resultados historicos importantes

### MNIST / digits

- FP32/BF16: aproximadamente `97.2%`.
- QPP LS/P-/P+: aproximadamente `95.7% - 95.8%`.
- Evidencia:

```text
outputs/qpp_mnist_digits_analysis/
```

### Qwen2.5-0.5B

Full all-linears rank256 trained:

- aceptados: `112/168`;
- BF16 PPL: `4.6657`;
- QPP PPL: `5.1353`;
- Delta PPL: `+0.4696`;
- ahorro total: `17.56%`;
- slowdown: `1.86x`;
- generacion mala.

### Qwen2-1.5B

Full all-linears rank256 trained:

- aceptados: `140/196`;
- BF16 PPL: `4.0494`;
- QPP PPL: `4.4944`;
- Delta PPL: `+0.4449`;
- ahorro total: `31.95%`;
- slowdown: `2.02x`;
- generacion mala/repetitiva.

### Qwen3-4B

Full direct+vectorized:

- aceptados: `98/252`;
- PPL: `4.4106`;
- Delta PPL: `+0.4919`;
- ahorro total: `24.74%`;
- generacion: `29.55 tok/s`;
- problema: repeticion severa.

Attention-only direct+vectorized:

- modulos: `66`;
- PPL: `3.9233`;
- Delta PPL: `+0.0046`;
- ahorro total: `8.25%`;
- generacion: `26.53 tok/s`;
- calidad smoke: coherente.

### Qwen2-7B

Carga BF16 local completa, pero no se pudo evaluar full en 16 GB VRAM:

```text
CUDA out of memory. Tried to allocate 298 MiB.
GPU total 15.57 GiB, free 152 MiB.
```

Para 7B hace falta offload, shards, baseline 8-bit/4-bit o kernel QPP mas eficiente.

## Por que full QPP repite

Hipotesis basada en experimentos:

- PPL local no captura bien la degradacion conversacional.
- MLP comprimido puede preservar PPL corto pero alterar dinamica de logits durante generacion autoregresiva.
- Capas tardias toleran mas compresion que capas tempranas, pero algunos MLP siguen introduciendo patrones repetitivos.
- El selector greedy por PPL debe complementarse con gate de generacion/repeticion.

## Siguiente trabajo recomendado

Prioridad 1: selector de calidad con gate de repeticion.

- Medir prompts multiples, no solo uno.
- Calcular repeticion de n-gramas.
- Rechazar modulos que mantengan PPL pero disparen repeticion.
- Rehacer seleccion para MLP con ese gate.

Prioridad 2: MLP selectivo de bajo riesgo.

- Empezar por capas tardias que dieron deltas negativos o casi cero.
- Probar perfiles por capa completa, no lineal individual.
- Mantener attention-only como baseline competitivo.

Prioridad 3: kernel Triton/CUDA fusionado.

- El fast-path PyTorch ya supera BF16 en generacion corta para attention-only.
- Para batch grande o full MLP, conviene fusionar gather + basis/anchors + acumulacion.
- Evitar `basis` por modulo o compartirlo por shape.

Prioridad 4: empaquetado recargable.

- Crear loader formal `load_qpp_artifact(model, artifact_path)`.
- Exponer API limpia para inferencia.
- Documentar que el artefacto referencia pesos BF16 base.

## Criterio de exito siguiente

Un siguiente resultado realmente fuerte seria:

```text
Qwen3-4B
ahorro >= 15%
PPL delta <= +0.05
tok/s >= BF16
generacion smoke coherente en varios prompts
sin repeticion evidente
```

El perfil actual ya cumple velocidad/calidad smoke, pero solo con `8.25%` de ahorro.

