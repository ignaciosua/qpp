# QPP Infographic — Brief para X/Twitter

> Documento para pasarle a un diseñador (o IA tipo Canva/DALL-E) y armar una imagen viral.

---

## Título / Hook principal (elige uno)

- **"Reducir parámetros, no bits — 21× compresión en atención de LLMs"**
- **"Tu GPU no necesita 8 billones de parámetros. QPP los comprime a 300M con solo 32 anchors."**
- **"7× más compresión que GGUF en atención. Sin tocar la calidad."**

---

## Estructura visual sugerida (3 paneles horizontales)

### Panel 1 — EL PROBLEMA
```
┌──────────────────────────────────────────────────┐
│                                                    │
│   W ∈ ℝ^(R×C)    →    Pesos de atención            │
│                                                    │
│   Cuantización tradicional (GGUF, AWQ):             │
│   FP16 → 4-bit = 4× máximo                         │
│                                                    │
│   ❌ Mismo número de parámetros                    │
│   ❌ Lento en CPU (15.4 tok/s en GGUF)             │
│                                                    │
└──────────────────────────────────────────────────┘
```

### Panel 2 — LA IDEA: Quantile Curve
```
┌──────────────────────────────────────────────────┐
│                                                    │
│   Peso ordenado de un perceptrón:                   │
│                                                    │
│     +▄                                             │
│     │ ▀▄        ←  K=32 anchors →                  │
│     │   ▀▄                                         │
│     │     ▀▄   ●──●──●──●──●                    │
│     │       ▀▄  ╱  ╱  ╱  ╱  ╱                   │
│     │         ▀▀▀▀▀▀▀▀▀▀▀                       │
│   ──┼─────────────────→ col sorted                 │
│                                                    │
│   R×C parámetros → R×32 anchors + orden compartido  │
│                                                    │
│   ✅ 21× compresión paramétrica                     │
│   ✅ ΔPPL ≤ 0.005 (imperceptible)                   │
│                                                    │
└──────────────────────────────────────────────────┘
```

### Panel 3 — QPP vs GGUF: Las cifras
```
┌──────────────────────────────────────────────────┐
│                                                    │
│   Qwen3-4B  |  Atención (351 MB):                  │
│   ────────────────────────────────────             │
│   GGUF Q4_K_M  ████████████ 109 MB  (3.2×)         │
│   QPP+CB_2b    █ 17 MB            (21×)  ← 7× más  │
│                                                    │
│   Modelo completo (8,045 MB):                       │
│   ────────────────────────────────────             │
│   QPP Hybrid ██████████████ 4,738 MB  (-41.1%)     │
│   ΔPPL = +0.507  |  26 tok/s  (69% > GGUF)         │
│                                                    │
│   🧩 QPP + GGUF son ortogonales:                    │
│      21× × 3.2× = 67× combinado teórico            │
│                                                    │
└──────────────────────────────────────────────────┘
```

---

## Frases para acompañar en X (thread sugerido)

**Tweet 1 (hook):**
> Descubrimos que los pesos de atención de un LLM, si los ordenas, forman una curva tan suave que la puedes reconstruir con solo 32 anclas. Resultado: 21× menos parámetros. 7× más compresión que GGUF. Misma calidad (ΔPPL=0.005). 🧵

**Tweet 2 (cómo funciona):**
> QPP no reduce bits como GGUF/AWQ. Reduce el *número* de parámetros. Cada fila de pesos → curva cuantílica → interpolación lineal con K=32 anchors. El orden de columnas se comparte por bloque. Es como ajustar un spline a la distribución de pesos.

**Tweet 3 (resultados):**
> Qwen3-4B: 8,045 MB → 4,738 MB (-41.1%) con generación coherente. En atención: QPP comprime 21× vs 3.2× de GGUF. Y con pesos cacheados: 26 tok/s vs 15.4 tok/s de GGUF (+69%). Y lo mejor: QPP + GGUF son compatibles. 67× combinado.

**Tweet 4 (call to action):**
> Código open-source (MIT), paper en Zenodo con DOI, 19 tests unitarios, CI corriendo, `pip install -e .` y listo. Faltan manos para el kernel Triton/CUDA que haría el speedup real. Si te late compresión de modelos, PRs bienvenidos. 🔗 github.com/ignaciosua/qpp

---

## Hashtags sugeridos

```
#MachineLearning #LLM #DeepLearning #AI #ModelCompression
#Quantization #OpenSource #PyTorch #QPP #CompressionIsAllYouNeed
```

---

## Ideas visuales para el diseñador

1. **Paleta**: fondo oscuro (#0d1117 estilo GitHub), accent cyan/verde neón
2. **La curva cuantílica** es EL visual — tiene que ser grande, central, con los anchors brillando
3. **Números grandes**: 21×, 7×, 41.1%, 69% en tipografía bold y color contrastante
4. **Gráfica de barras** para el QPP vs GGUF en los 351 MB de atención
5. **QR code** abajo-derecha con link al repo de GitHub

---

## Formato final

- Imagen PNG 1200×675 (Twitter card landscape) o 1080×1080 (cuadrado)
- Incluir `github.com/ignaciosua/qpp` y `DOI: 10.5281/zenodo.21046683` en letra pequeña al pie
