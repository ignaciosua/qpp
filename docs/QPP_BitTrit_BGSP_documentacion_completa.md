# BitTrit-QPP / BGSP con Función Cuantílica

**Documentación técnica completa del enfoque trabajado**  
**Versión:** 0.2  
**Propósito:** dejar una especificación clara para que una IA, investigador o implementador pueda continuar experimentando con el enfoque completo: pesos ternarios/BitTrit, función cuantílica por tramos, puente central, escaladores, acumuladores, bit-planes, early-exit y posible integración con BGSP.

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Problema que intenta resolver](#2-problema-que-intenta-resolver)
3. [Observación central: los pesos ordenados dibujan una función cuantil](#3-observación-central-los-pesos-ordenados-dibujan-una-función-cuantil)
4. [Definiciones básicas](#4-definiciones-básicas)
5. [Modelo QPP: Quantile Piecewise Perceptron](#5-modelo-qpp-quantile-piecewise-perceptron)
6. [Representación de tres segmentos con puente central](#6-representación-de-tres-segmentos-con-puente-central)
7. [Ajuste por Least Squares](#7-ajuste-por-least-squares)
8. [Backprop sobre la misma función](#8-backprop-sobre-la-misma-función)
9. [Trits, BitTrit y BGSP](#9-trits-bittrit-y-bgsp)
10. [Forward rápido con trits y bit-planes](#10-forward-rápido-con-trits-y-bit-planes)
11. [Función cuantil + acumuladores](#11-función-cuantil--acumuladores)
12. [Cuantización del cuantil por buckets](#12-cuantización-del-cuantil-por-buckets)
13. [Early-exit por cotas](#13-early-exit-por-cotas)
14. [Backward y entrenamiento](#14-backward-y-entrenamiento)
15. [Resultados experimentales obtenidos](#15-resultados-experimentales-obtenidos)
16. [Arquitectura recomendada actual](#16-arquitectura-recomendada-actual)
17. [Protocolos de experimentación](#17-protocolos-de-experimentación)
18. [Pseudocódigo principal](#18-pseudocódigo-principal)
19. [Métricas que deben reportarse](#19-métricas-que-deben-reportarse)
20. [Errores que ya detectamos y no deben repetirse](#20-errores-que-ya-detectamos-y-no-deben-repetirse)
21. [Hipótesis abiertas](#21-hipótesis-abiertas)
22. [Roadmap de investigación](#22-roadmap-de-investigación)
23. [Prompt operativo para otra IA](#23-prompt-operativo-para-otra-ia)
24. [Conclusión](#24-conclusión)

---

# 1. Resumen ejecutivo

Este proyecto propone representar un perceptrón no como una lista de pesos independientes, sino como una **función cuantil por tramos**.

La observación inicial fue:

> Si se ordenan los pesos entrenados de un perceptrón de menor a mayor, normalmente forman una curva suave, monotónica y estructurada.

Esto sugiere que los pesos no son completamente aleatorios. Hay redundancia geométrica.

El enfoque se llama aquí:

```text
QPP = Quantile Piecewise Perceptron
```

La versión más estable hasta ahora usa:

```text
rama negativa + puente central + rama positiva
```

con parámetros:

```text
a-, m-, a+, m+, bias
```

Opcionalmente se agregan:

```text
P-, P+
```

para calibrar por separado la rama negativa y positiva.

El propósito final **no es solo comprimir pesos**, sino explorar si esta representación permite transformar el producto punto tradicional:

```math
y = \sum_i w_i x_i + b
```

en una evaluación por **acumuladores agrupados**:

```math
y \approx c_1 A^- + c_2 B^- + c_3 A^0 + c_4 B^0 + c_5 A^+ + c_6 B^+ + b
```

Donde los acumuladores son sumas de entradas agrupadas por región cuantil.

La integración con **BitTrit / BGSP** es natural porque los pesos pueden ser ternarios:

```text
w_i direction ∈ {-1, 0, +1}
```

y la magnitud puede provenir de la función cuantil.

---

# 2. Problema que intenta resolver

Un perceptrón tradicional calcula:

```math
y = \sum_{i=1}^{n} w_i x_i + b
```

Esto requiere, en forma directa:

```text
n multiplicaciones peso × entrada
n sumas
n pesos almacenados
```

En redes grandes esto implica:

- Mucha memoria.
- Muchos accesos a memoria.
- Muchas multiplicaciones.
- Difícil compresión sin pérdida.
- Forward y backward densos.

La pregunta principal es:

> ¿Podemos reemplazar la lista de pesos por una función de pocos parámetros y además evaluar el perceptrón usando menos cómputo?

---

# 3. Observación central: los pesos ordenados dibujan una función cuantil

Dado un vector de pesos entrenados:

```math
W = (w_1,w_2,...,w_n)
```

Ordenamos:

```math
w_{(1)} \le w_{(2)} \le ... \le w_{(n)}
```

Asignamos a cada peso una posición cuantil:

```math
q_i = \frac{i}{n-1}, \quad q_i \in [0,1]
```

Entonces graficamos:

```text
q_i  vs  w_(i)
```

En MNIST 8×8 se observó que las curvas son usualmente suaves y monotónicas.

Esto llevó a la hipótesis:

> La distribución ordenada de pesos de una neurona puede aproximarse con una función continua de pocos parámetros.

---

# 4. Definiciones básicas

## 4.1 Perceptrón

```math
y = W x + b
```

Para una sola neurona:

```math
y = \sum_i w_i x_i + b
```

## 4.2 Función cuantil de pesos

La función cuantil aproximada es:

```math
Q(q) \approx w_{(i)}
```

con:

```math
q = \frac{i}{n-1}
```

## 4.3 Trit

Un trit es un valor ternario:

```math
t_i \in \{-1,0,+1\}
```

Interpretación:

```text
-1 = conexión negativa
 0 = conexión apagada o zona central
+1 = conexión positiva
```

## 4.4 Bit-plane

Si una entrada se representa en bits:

```math
x_i = \sum_{p=0}^{B-1} 2^p b_{i,p}
```

con:

```math
b_{i,p} \in \{0,1\}
```

entonces el producto con trits puede evaluarse por máscaras y `popcount`.

## 4.5 BGSP

En esta documentación, BGSP se entiende como una familia de enfoques donde el producto punto se simplifica usando signos, trits, agrupaciones o escaladores.

Forma base simplificada:

```math
y = P \sum_i \operatorname{sign}(w_i)x_i + b
```

La propuesta aquí extiende esa idea con una función cuantil:

```math
w_i \approx t_i Q(q_i)
```

---

# 5. Modelo QPP: Quantile Piecewise Perceptron

El QPP propone que la neurona tenga dos partes:

1. **Estructura discreta:** trits o máscaras que indican región/dirección.
2. **Función cuantil:** curva de pocos parámetros que define magnitudes.

Representación conceptual:

```text
entrada x
  ↓
acumuladores agrupados
  ↓
función cuantil Q(q)
  ↓
salida y
```

En lugar de:

```text
entrada x
  ↓
muchos pesos independientes
  ↓
producto punto completo
  ↓
salida y
```

---

# 6. Representación de tres segmentos con puente central

La representación más estable hasta ahora fue:

```text
1. Rama negativa
2. Puente central
3. Rama positiva
```

## 6.1 Rama negativa

Aproxima los pesos negativos ordenados.

Parámetros:

```text
a- = intercepto/base negativa
m- = pendiente negativa
```

## 6.2 Rama positiva

Aproxima los pesos positivos ordenados.

Parámetros:

```text
a+ = intercepto/base positiva
m+ = pendiente positiva
```

## 6.3 Puente central

El centro **no tiene parámetros independientes**.

Se obtiene por interpolación entre la frontera de la rama negativa y la frontera de la rama positiva.

Esto evita discontinuidades y reduce parámetros.

## 6.4 Parámetros mínimos

```text
a-, m-, a+, m+, bias
```

Total:

```text
5 parámetros por perceptrón
```

en lugar de:

```text
n pesos + bias
```

## 6.5 Con escaladores

Se pueden agregar:

```text
P-, P+
```

Entonces los parámetros serían:

```text
a-, m-, a+, m+, P-, P+, bias
```

Total:

```text
7 parámetros por perceptrón
```

---

# 7. Ajuste por Least Squares

## 7.1 Procedimiento

1. Entrenar modelo full precision.
2. Tomar pesos de cada perceptrón.
3. Ordenar pesos.
4. Identificar región negativa, centro y positiva.
5. Construir base de 3 segmentos.
6. Ajustar parámetros por mínimos cuadrados.
7. Reconstruir pesos aproximados.
8. Medir accuracy, MAE de logit y R².

## 7.2 Por qué funciona

Least Squares resuelve directamente:

```math
\theta^* = \arg\min_\theta ||W - \hat{W}(\theta)||^2
```

Es decir, busca la mejor aproximación geométrica de los pesos bajo la familia de funciones elegida.

## 7.3 Resultado observado

En MNIST 8×8:

```text
Full precision:     97.11% accuracy
3 segmentos LS:     96.22% accuracy
```

MAE del logit:

```text
LS: 0.6824 aproximadamente
```

Conclusión:

> LS es la mejor base actual para reconstrucción post-training.

---

# 8. Backprop sobre la misma función

También se probó optimizar los mismos parámetros por backprop.

## 8.1 Estructura idéntica

Se mantuvo exactamente la misma función:

```text
rama negativa + puente central + rama positiva
```

Mismos parámetros:

```text
a-, m-, a+, m+, bias
```

## 8.2 Diferencia

LS minimiza error de reconstrucción.

Backprop minimiza clasificación:

```math
\mathcal{L}=\operatorname{CrossEntropy}(y, \hat{y})
```

## 8.3 Resultado observado

```text
Full precision:             97.11%
3 segmentos LS:             96.22%
3 segmentos backprop:       96.67%
```

Pero backprop empeoró la reconstrucción del logit:

```text
MAE LS:       ~0.6824
MAE backprop: ~1.4834
```

Conclusión:

> Backprop puede mejorar ligeramente accuracy, pero deforma la curva y descalibra logits.

---

# 9. Trits, BitTrit y BGSP

## 9.1 Trits como direcciones

El enfoque con trits usa:

```math
t_i \in \{-1,0,+1\}
```

El producto tradicional:

```math
\sum_i w_i x_i
```

se aproxima por:

```math
\sum_i t_i Q(q_i) x_i
```

Aquí:

- `t_i` determina dirección o apagado.
- `Q(q_i)` determina magnitud aproximada según posición cuantil.

## 9.2 BitTrit

BitTrit combina:

- Entradas descompuestas en bit-planes.
- Pesos ternarios.
- Máscaras binarias para positivo, negativo y cero.
- Posible función cuantil para magnitud.

## 9.3 BGSP + función cuantil

BGSP simple:

```math
y = P \sum_i t_i x_i + b
```

QPP/BGSP cuantílico:

```math
y = \sum_i t_i Q(q_i)x_i + b
```

Con escaladores:

```math
y = P^- \sum_{i \in neg} Q^-(q_i)x_i + P^+ \sum_{i \in pos} Q^+(q_i)x_i + b
```

---

# 10. Forward rápido con trits y bit-planes

Si las entradas se representan por bit-planes:

```math
x_i = \sum_{p=0}^{B-1} 2^p b_{i,p}
```

con pesos ternarios:

```math
t_i \in \{-1,0,+1\}
```

entonces:

```math
y = \sum_p 2^p \left( \sum_{i:t_i=+1} b_{i,p} - \sum_{i:t_i=-1} b_{i,p} \right)
```

Esto puede implementarse con máscaras:

```text
pos_mask = t == +1
neg_mask = t == -1
```

Para cada bit-plane:

```text
pos_count = popcount(x_bits[p] & pos_mask)
neg_count = popcount(x_bits[p] & neg_mask)
y += 2^p * (pos_count - neg_count)
```

Esto elimina multiplicaciones peso × entrada si solo hay trits puros.

Con función cuantil, hay magnitudes. Para mantener eficiencia se deben agrupar magnitudes por buckets.

---

# 11. Función cuantil + acumuladores

La pregunta crítica es:

> ¿Cómo usar la función cuantil sin reconstruir cada peso?

La respuesta propuesta:

No convertir la entrada en parámetros del cuantil.  
Los parámetros del cuantil pertenecen al perceptrón.

La entrada se transforma en **acumuladores** compatibles con la función.

## 11.1 Acumuladores básicos

Para la rama negativa:

```math
A^- = \sum_{i \in neg} x_i
```

```math
B^- = \sum_{i \in neg} r_i x_i
```

Para la rama positiva:

```math
A^+ = \sum_{i \in pos} x_i
```

```math
B^+ = \sum_{i \in pos} r_i x_i
```

Opcionalmente para centro:

```math
A^0 = \sum_{i \in center} x_i
```

```math
B^0 = \sum_{i \in center} r_i x_i
```

## 11.2 Salida con acumuladores

```math
y \approx c_1 A^- + c_2 B^- + c_3 A^0 + c_4 B^0 + c_5 A^+ + c_6 B^+ + b
```

Los coeficientes `c_k` dependen de:

```text
a-, m-, a+, m+, P-, P+
```

Esto reemplaza muchos productos `w_i x_i` por pocas sumas agrupadas y pocas combinaciones finales.

---

# 12. Cuantización del cuantil por buckets

El problema de `B = Σ r_i x_i` es que `r_i` puede ser distinto para cada entrada.

Para hacerlo rápido:

```text
cuantizar r_i en buckets
```

Ejemplo:

```text
r_i ∈ {0, 0.25, 0.5, 0.75, 1}
```

Entonces:

```math
\sum_i r_i x_i \approx 0S_0 + 0.25S_1 + 0.5S_2 + 0.75S_3 + 1S_4
```

Donde:

```math
S_k = \sum_{i: r_i \in bucket_k} x_i
```

Esto permite calcular cada `S_k` con máscaras.

En BitTrit, cada `S_k` puede calcularse con `popcount` por bit-plane.

---

# 13. Early-exit por cotas

También se probó una idea de terminación temprana.

## 13.1 Idea

Calcular primero los términos más importantes.

Si el resto ya no puede cambiar la decisión, detener el cálculo.

## 13.2 Cota general

Suma parcial:

```math
S_k = \sum_{i=1}^{k} w_{(i)}x_{(i)}
```

Resto máximo:

```math
R_k = \sum_{i=k+1}^{n}|w_{(i)}x_{(i)}|
```

Si:

```math
|S_k| > R_k
```

entonces el resto no puede cambiar el signo.

En clasificación multiclase:

```text
parar si lower_bound(clase ganadora) > upper_bound(todas las demás)
```

## 13.3 Resultado observado

En MNIST 8×8:

```text
orden fijo por |w|:
~57.9% ahorro promedio de términos
sin cambiar accuracy
```

Esto prueba que hay potencial de ahorro, aunque falta demostrar speedup real en GPU.

## 13.4 En BitTrit

El orden natural no es por magnitud de pesos, sino por bit-plane:

```text
MSB → LSB
```

Los bits más significativos dominan.

Forward early-exit:

```text
for bitplane from MSB to LSB:
    delta = popcount(pos) - popcount(neg)
    S += scale * delta
    if bounds guarantee decision:
        break
```

---

# 14. Backward y entrenamiento

## 14.1 Entrenamiento post-training recomendado

El método más estable actualmente:

```text
1. Entrenar full precision
2. Extraer pesos
3. Ternarizar / ordenar
4. Ajustar QPP por LS
5. Opcional: fine-tune de bias, P-, P+
```

## 14.2 Entrenamiento directo aún no estable

Se intentó aprender trits y curva directamente.

Resultado:

- Soft funcionó razonablemente.
- Hard cayó mucho.

Conclusión:

> Hace falta mejor STE, annealing o entrenamiento híbrido.

## 14.3 Entrenamiento con STE propuesto

Latentes reales:

```math
z_i \in \mathbb{R}
```

Trit duro en forward:

```math
t_i = \operatorname{ternary}(z_i)
```

Backward aproximado:

```math
\frac{\partial L}{\partial z_i} \approx \frac{\partial L}{\partial t_i}
```

Este enfoque aún debe probarse de forma robusta.

## 14.4 Backward con acumuladores

Si forward usa acumuladores:

```math
y = c_1A^- + c_2B^- + ... + b
```

Entonces gradientes:

```math
\frac{\partial L}{\partial c_k} = \frac{\partial L}{\partial y} A_k
```

```math
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y} \cdot c_{region(i)}
```

Para buckets:

```math
\frac{\partial L}{\partial S_k} = \frac{\partial L}{\partial y} \cdot \alpha_k
```

Esto podría hacer backward más estructurado.

---

# 15. Resultados experimentales obtenidos

## 15.1 MNIST 8×8

Modelo base:

```text
One-vs-rest logistic regression / perceptrones lineales
10 clases
64 entradas
```

Resultados aproximados:

| Variante | Accuracy | Comentario |
|---|---:|---|
| Full precision | 97.11% | Referencia |
| Trits + 1 escalar | 89.56% | Baja bastante |
| Trits + 3 escalares planos | 89.56% | No mejora accuracy |
| Trits + 3 rectas con pendiente | 95.78% | Gran mejora |
| 3 rectas + P global | 96.00% | Mejora leve |
| 3 rectas + P+/P- | 96.44% | Mejor calibración por rama |
| Puente central LS | 96.22% | Mejor reconstrucción |
| Puente central backprop | 96.67% | Mejor accuracy aproximada |
| Trits+curva soft | 96.44% | Funciona suave |
| Trits+curva hard | 88.44% | Discretización inestable |

## 15.2 Early-exit

| Método | Accuracy | Ahorro promedio |
|---|---:|---:|
| Producto completo | 97.11% | 0% |
| Orden fijo por \\|w\\| con cota segura | 97.11% | ~57.9% |
| Orden ideal por \\|w·x\\| | 97.11% | ~65.7% |

---

# 16. Arquitectura recomendada actual

La versión recomendada hoy:

```text
Full precision inicial
  ↓
Ordenar pesos
  ↓
Ternarizar / identificar regiones
  ↓
Ajustar 3 segmentos con puente central por LS
  ↓
Agregar P-, P+ opcionalmente
  ↓
Explorar forward con acumuladores
```

No se recomienda aún:

```text
entrenar trits hard desde cero
```

porque todavía fue inestable.

---

# 17. Protocolos de experimentación

## 17.1 Protocolo A: compresión post-training

1. Entrenar baseline full precision.
2. Medir accuracy.
3. Extraer pesos.
4. Ordenar por perceptrón.
5. Ajustar 3 segmentos LS.
6. Reconstruir pesos.
7. Medir:
   - Accuracy.
   - MAE logit.
   - R² logit.
   - Diferencia de predicción contra full.
8. Agregar P+/P-.
9. Repetir métricas.

## 17.2 Protocolo B: acumuladores

1. Tomar QPP ajustado.
2. Crear regiones/buckets.
3. Precomputar máscaras.
4. Para cada entrada calcular acumuladores.
5. Calcular salida aproximada.
6. Comparar contra producto punto reconstruido.
7. Medir tiempo CPU.
8. Medir tiempo GPU.
9. Comparar contra baseline denso.

## 17.3 Protocolo C: BitTrit

1. Cuantizar entradas a bit-planes.
2. Convertir trits a máscaras `pos`, `neg`, `zero`.
3. Agrupar por buckets de cuantil.
4. Evaluar con `popcount`.
5. Agregar escalas de bit-plane.
6. Medir accuracy y tiempo.

## 17.4 Protocolo D: entrenamiento directo

1. Inicializar latentes `z_i`.
2. Forward con trits hard.
3. Backward con STE.
4. Curva QPP aprendible.
5. Regularizar suavidad/monotonicidad.
6. Comparar soft vs hard.
7. Medir estabilidad.

---

# 18. Pseudocódigo principal

## 18.1 Ajuste LS de 3 segmentos

```python
for neuron in layer:
    w = neuron.weights
    order = argsort(w)
    w_sorted = w[order]

    # Crear trits/regiones
    t = ternarize(w_sorted)
    neg_idx = where(t == -1)
    zero_idx = where(t == 0)
    pos_idx = where(t == +1)

    # Construir matriz de base B con 4 columnas:
    # a-, m-, a+, m+
    B = build_3segment_bridge_basis(t, neg_idx, zero_idx, pos_idx)

    # Resolver mínimos cuadrados
    theta = lstsq(B, w_sorted)

    # Reconstruir
    w_hat_sorted = B @ theta
    w_hat[order] = w_hat_sorted
```

## 18.2 Forward con acumuladores

```python
# Precomputado por perceptrón:
neg_mask
center_mask
pos_mask
r_bucket_masks
coefficients c1..c6

for sample x:
    Aneg = sum(x[neg_mask])
    Bneg = sum(r[neg_mask] * x[neg_mask])

    Apos = sum(x[pos_mask])
    Bpos = sum(r[pos_mask] * x[pos_mask])

    A0 = sum(x[center_mask])
    B0 = sum(r[center_mask] * x[center_mask])

    y = c1*Aneg + c2*Bneg + c3*A0 + c4*B0 + c5*Apos + c6*Bpos + bias
```

## 18.3 Forward BitTrit por bit-planes

```python
for bitplane p from MSB to LSB:
    x_bits = packed_input[p]

    for bucket k:
        pos_count = popcount(x_bits & pos_mask[k])
        neg_count = popcount(x_bits & neg_mask[k])

        delta[k] += scale[p] * (pos_count - neg_count)

# combinar buckets con coeficientes cuantílicos
logit = sum_k coeff[k] * delta[k] + bias
```

## 18.4 Early-exit BitTrit

```python
S = bias
remaining_bound = compute_max_remaining()

for p in bitplanes_from_MSB_to_LSB:
    S += contribution_from_bitplane(p)
    remaining_bound -= max_possible_contribution(p)

    if decision_is_guaranteed(S, remaining_bound):
        break
```

---

# 19. Métricas que deben reportarse

Toda prueba debe reportar:

## Accuracy

```text
Top-1 accuracy
```

## Coincidencia con full

```text
mean(pred_approx == pred_full)
```

## MAE de logit

```math
MAE = mean(|logit_full - logit_approx|)
```

## R² de logit

```math
R^2(logit_full, logit_approx)
```

## Costo computacional

- Número de multiplicaciones.
- Número de sumas.
- Número de popcounts.
- Memoria de pesos.
- Memoria de máscaras.
- Tiempo CPU.
- Tiempo GPU.

## Estabilidad hard vs soft

Reportar siempre:

```text
soft accuracy
hard accuracy
caída soft → hard
```

---

# 20. Errores que ya detectamos y no deben repetirse

## Error 1: comparar funciones distintas

Una versión soft puede crear más de 3 segmentos reales.

La comparación justa debe usar la misma función.

## Error 2: confundir compresión con aceleración

Reconstruir pesos comprimidos y luego hacer `x @ W` no acelera.

Para acelerar hay que usar acumuladores o bitpacking.

## Error 3: entrenar trits soft y asumir que hard funcionará

No necesariamente.

Se observó caída fuerte al discretizar.

## Error 4: dejar que backprop deforme todo sin control

Backprop puede mejorar accuracy pero destruir calibración de logits.

Usar distillation o regularización si se necesita conservar logit.

## Error 5: no medir contra baseline real

Siempre comparar contra:

```text
full precision baseline
producto punto denso
implementación CPU/GPU real
```

---

# 21. Hipótesis abiertas

1. P+/P- puede recuperar precisión sin deformar la curva.
2. Buckets de cuantil pueden reducir operaciones reales.
3. BitTrit + popcount + buckets puede ser más rápido que denso en CPU/ASIC/FPGA.
4. GPU necesita diseño por bloques para evitar divergencia.
5. Early-exit por bit-plane puede ahorrar cómputo en inferencia.
6. LS incremental podría permitir entrenamiento más rápido.
7. Entrenar QPP desde cero requiere STE mejor diseñado.
8. Función cuantil podría actuar como prior estructural que mejora generalización.

---

# 22. Roadmap de investigación

## Etapa 1: Validación geométrica

Estado: hecho.

- Pesos ordenados forman curvas.
- 3 segmentos aproximan bien.
- LS funciona.

## Etapa 2: Calibración

Estado: parcialmente hecho.

- P global probado.
- P+/P- probado.
- Falta fine-tuning sistemático.

## Etapa 3: Cómputo con acumuladores

Estado: pendiente.

- Implementar forward sin reconstruir pesos.
- Comparar tiempo real.

## Etapa 4: BitTrit

Estado: pendiente.

- Implementar bit-planes.
- Implementar máscaras por bucket.
- Usar popcount.

## Etapa 5: GPU/CUDA

Estado: pendiente.

- Diseñar kernels por bloque.
- Evitar divergencia.
- Medir speedup real.

## Etapa 6: Entrenamiento

Estado: abierto.

- STE hard trits.
- Backward estructurado.
- Full refresh ocasional.

## Etapa 7: Escalamiento

Estado: pendiente.

- CIFAR.
- MLP más grande.
- Transformers pequeños.
- Modelos tipo GPT.

---

# 23. Prompt operativo para otra IA

Usa el siguiente prompt si se quiere que otra IA continúe el trabajo.

```text
Quiero que investigues e implementes el enfoque BitTrit-QPP/BGSP con función cuantil.

Resumen técnico:

Tenemos una observación: al ordenar los pesos de un perceptrón entrenado, estos forman una curva suave tipo función cuantil. Queremos representar cada perceptrón no como pesos independientes, sino como una función por tramos:

- rama negativa
- puente central por interpolación
- rama positiva

Parámetros mínimos:

a-, m-, a+, m+, bias

Variante extendida:

a-, m-, a+, m+, P-, P+, bias

Los trits indican dirección:

t_i ∈ {-1, 0, +1}

La magnitud se obtiene de la función cuantil Q(q_i).

Objetivos:

1. Reproducir MNIST 8×8 con full precision baseline.
2. Ajustar 3 segmentos por Least Squares.
3. Comparar contra backprop usando exactamente la misma función.
4. Probar P+ y P-.
5. Implementar forward con acumuladores sin reconstruir todos los pesos.
6. Implementar versión BitTrit con entradas por bit-planes y pesos ternarios.
7. Usar máscaras + popcount por bucket de cuantil.
8. Medir accuracy, MAE logit, R², coincidencia con full, operaciones y tiempo real.
9. Comparar contra producto punto denso CPU/GPU.
10. Probar early-exit por bit-plane usando cotas.

Restricciones importantes:

- No compares funciones distintas. Si comparas LS vs backprop, debe ser la misma función de 3 segmentos.
- No afirmes aceleración si solo reconstruyes pesos y haces x @ W.
- Para acelerar, debes usar acumuladores, máscaras, buckets o popcount.
- Reporta soft y hard por separado si usas trits aprendibles.
- Si hard cae, documenta la caída.

Resultados previos esperados en MNIST 8×8:

Full precision ≈ 97.11%
3 segmentos LS ≈ 96.22%
3 segmentos backprop ≈ 96.67%
P+/P- alrededor de ≈ 96.44%
Trits soft ≈ 96.44%
Trits hard puede caer ≈ 88.44% si no se entrena bien.

El objetivo final no es solo comprimir, sino convertir la función cuantil en una forma de cómputo rápido mediante acumuladores agrupados y bitpacking.
```

---

# 24. Conclusión

El proyecto BitTrit-QPP/BGSP propone una reinterpretación del perceptrón.

En lugar de verlo como:

```text
lista de pesos independientes
```

se ve como:

```text
estructura ternaria + función cuantil suave + pocos parámetros de calibración
```

Lo ya demostrado:

- Los pesos ordenados forman curvas suaves.
- 3 segmentos aproximan bien.
- LS es excelente para reconstrucción.
- Backprop puede mejorar un poco accuracy.
- P+/P- ayudan a calibrar.
- Trits hard todavía son inestables.
- Early-exit tiene potencial de ahorro.

Lo más importante:

> La compresión no basta. La siguiente fase debe demostrar cómputo rápido real usando acumuladores, buckets, bit-planes, máscaras y popcount.

La arquitectura final buscada es:

```text
entrada
  ↓
bit-planes / acumuladores
  ↓
buckets cuantílicos
  ↓
función QPP
  ↓
salida
```

Si esto funciona, el resultado sería un perceptrón más compacto, más interpretable y potencialmente más eficiente para inferencia y entrenamiento.

---

# Apéndice A: Nombres sugeridos

- QPP: Quantile Piecewise Perceptron.
- BitTrit-QPP: versión con bit-planes y trits.
- Q-BGSP: BGSP con función cuantil.
- QTP: Quantile Trit Perceptron.
- QC-BGSP: Quantile Curve BGSP.

---

# Apéndice B: Configuración mínima para pruebas

Dataset inicial recomendado:

```text
sklearn digits / MNIST 8×8
```

Modelo inicial:

```text
10 perceptrones one-vs-rest
64 entradas
```

Baseline:

```text
LogisticRegression o MLP lineal
```

Semilla:

```text
random_state = 42
```

Métricas mínimas:

```text
accuracy
MAE logit
R² logit
coincidencia con full
número de parámetros
número estimado de operaciones
```

---

# Apéndice C: Tabla de variantes

| Variante | Parámetros | Objetivo | Estado |
|---|---:|---|---|
| Full precision | n+bias | referencia | hecho |
| Trits + 1 escalar | trits + P | compresión simple | probado |
| Trits + 3 escalares | trits + 3P | secciones planas | probado |
| Trits + 3 rectas | trits + pendientes | mejor curva | probado |
| Puente central LS | 5 params | reconstrucción | mejor base |
| Puente central backprop | 5 params | clasificación | mejora accuracy |
| P+/P- | 7 params | calibración | prometedor |
| Trits soft aprendibles | muchos logits | entrenamiento suave | funciona soft |
| Trits hard aprendibles | trits discretos | arquitectura final | inestable |
| Acumuladores | pocos sums | speedup | pendiente |
| BitTrit popcount | masks + bitplanes | hardware rápido | pendiente |

---

# Apéndice D: Criterio de éxito

El enfoque debe considerarse exitoso solo si cumple al menos uno de estos criterios:

1. Mantener accuracy cercana a full precision con mucha menos memoria.
2. Reducir operaciones reales medidas en CPU/GPU.
3. Permitir forward más rápido con bitpacking/popcount.
4. Mantener backward entrenable sin colapso.
5. Escalar a modelos más grandes sin perder estabilidad.

Criterio ideal:

```text
accuracy ≈ full precision
+
menos memoria
+
menos tiempo real
+
implementación compatible con hardware
```

---

**Fin del documento.**
