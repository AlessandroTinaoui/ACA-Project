# Cosa implementare adesso, in ordine

Questo documento riassume la roadmap implementativa per il progetto KAN su RISC-V/gem5, considerando lo stato attuale della repo `KAN-gem5` e le idee discusse: local active B-spline evaluation, LUT, quantizzazione, packed SIMD, custom instruction `KAN.DOT4` e scratchpad.

---

## 1. Misurazione pulita della sola inferenza

**Importanza:** altissima  
**Difficoltà:** bassa  
**Dove inserirla:** `src/main.c`, script gem5, eventualmente m5ops.

### Problema attuale

Oggi `src/main.c` misura tutto insieme:

- inizializzazione;
- loop sugli input;
- chiamate a `kan_infer(x)`;
- calcolo della funzione target;
- accumulo metriche;
- `printf`;
- terminazione programma.

Per un lavoro hardware-oriented, questo è troppo “sporco”: le statistiche gem5 includono anche lavoro che non appartiene direttamente all’inferenza KAN.

### Cosa fare

Separare meglio:

```text
setup
warm-up opzionale
reset stats gem5
loop inferenza KAN
dump stats gem5
calcolo/stampa metriche
```

Idealmente usare m5ops per fare:

```c
m5_reset_stats(0, 0);
/* inference loop */
m5_dump_stats(0, 0);
```

oppure, se non vuoi ancora usare m5ops, almeno ridurre al minimo `printf` e codice extra dentro la regione misurata.

### Perché è importante

Prima di ottimizzare devi sapere cosa stai misurando. Se non isoli il kernel, non sai se lo speedup viene dalla KAN, dalla riduzione delle stampe, dal parsing o dal codice di confronto.

---

## 2. Ottimizzazione locale delle B-spline: calcolare solo le basi attive

**Importanza:** altissima  
**Difficoltà:** media  
**Dove inserirla:** `src/bspline.c`.

### Stato attuale

Il commit di fix ha già migliorato `bspline_eval`: sei passato da Cox-de Boor ricorsivo a una versione iterativa dynamic-programming con `prev[]` e `curr[]`.

Oggi però la funzione calcola ancora tutte le basi di grado 3 e poi fa:

```c
for (int i = 0; i < num_control_points; ++i) {
    y += control_points[i] * prev[i];
}
```

Con `num_control_points = 15`, significa che per ogni edge combini 15 coefficienti.

### Osservazione matematica

Per una B-spline di grado `p`, in un dato intervallo `[t_k, t_{k+1})` sono attive al massimo `p + 1` basi.

Nel tuo caso:

```text
p = 3
p + 1 = 4
```

Quindi per una spline cubica servono solo 4 basi locali.

### Cosa implementare

Creare una nuova funzione, per esempio:

```c
float bspline_eval_local4(
    float x,
    const float *knots,
    const float *control_points,
    int degree,
    int num_control_points,
    int num_knots
);
```

che faccia:

```text
1. trova l'intervallo k tale che x cade in [t_k, t_{k+1})
2. calcola solo le 4 basi cubiche attive
3. carica solo i 4 coefficienti corrispondenti
4. restituisce la somma pesata
```

Forma finale:

```text
phi(x) = c0*b0(x) + c1*b1(x) + c2*b2(x) + c3*b3(x)
```

### Perché è importante

Questo è il ponte tra il codice attuale e tutte le ottimizzazioni hardware:

```text
4 basi attive
    ↓
4 coefficienti
    ↓
4 MAC
    ↓
packed int8
    ↓
KAN.DOT4
```

---

## 3. Cambiare layout dei coefficienti in `coeff4` per intervallo

**Importanza:** altissima  
**Difficoltà:** media  
**Dove inserirla:** `scripts/json_to_header.py` e header generato.

### Stato attuale

Oggi i coefficienti sono salvati come:

```c
KAN_LAYER_CONTROL_POINTS[layer][input][output][num_control_points]
```

Quindi il layout è pensato per la formulazione matematica completa, non per l’accesso locale hardware-friendly.

### Cosa fare

Generare una seconda rappresentazione dei coefficienti, organizzata per intervallo locale:

```c
float KAN_LAYER_COEFF4
    [KAN_NUM_LAYERS]
    [KAN_MAX_INPUT_DIM]
    [KAN_MAX_OUTPUT_DIM]
    [KAN_NUM_INTERVALS]
    [4];
```

Ogni entry contiene i 4 coefficienti attivi per quell’intervallo.

### Versione futura quantizzata

Dopo la versione float, passare a:

```c
uint32_t KAN_LAYER_COEFF4_Q8
    [KAN_NUM_LAYERS]
    [KAN_MAX_INPUT_DIM]
    [KAN_MAX_OUTPUT_DIM]
    [KAN_NUM_INTERVALS];
```

dove ogni `uint32_t` contiene:

```text
byte 0 = c0 int8
byte 1 = c1 int8
byte 2 = c2 int8
byte 3 = c3 int8
```

### Perché è importante

Così il kernel non deve più cercare o combinare 15 coefficienti. Per ogni intervallo carica direttamente il pacchetto locale:

```c
coeff4 = KAN_LAYER_COEFF4_Q8[layer][input][output][idx];
```

Questo abilita:

- memoria allineata;
- load a 32 bit;
- packed SIMD;
- custom instruction `KAN.DOT4`;
- scratchpad più compatto.

---

## 4. Aggiungere una LUT piccola per le basis spline

**Importanza:** molto alta  
**Difficoltà:** media  
**Dove inserirla:** `scripts/json_to_header.py`, nuovo `scripts/generate_basis_lut.py`, oppure direttamente `src/bspline.c`.

### Idea

Le basis spline locali dipendono dalla posizione di `x` dentro l’intervallo, non dall’edge.

I coefficienti cambiano per edge; le basis, invece, sono condivisibili.

Quindi puoi precomputare:

```c
basis_lut[NUM_FRAC_BINS][4]
```

Esempio:

```text
NUM_FRAC_BINS = 64, 128, 256
```

Per ogni valore frazionario locale `frac`, la LUT restituisce:

```text
[b0, b1, b2, b3]
```

### Runtime

Il kernel diventa:

```text
idx = intervallo in cui cade x
frac = posizione locale di x dentro l'intervallo
basis4 = basis_lut[frac]
coeff4 = coeff4_packed[edge][idx]
acc += dot4(coeff4, basis4)
```

### Perché è importante

Eviti di calcolare Cox-de Boor a runtime per ogni edge.

La LUT è piccola:

```text
256 entry * 4 valori * 1 byte = 1024 byte  se int8
256 entry * 4 valori * 2 byte = 2048 byte  se int16
```

Quindi può stare bene in cache, in scratchpad o in una piccola ROM dedicata.

---

## 5. Quantizzazione INT8/INT16

**Importanza:** altissima  
**Difficoltà:** medio-alta  
**Dove inserirla:** exporter/header + nuovo kernel C.

### Obiettivo

Trasformare la spline evaluation da float a fixed-point/integer.

Schema consigliato:

```text
coefficienti spline: int8
basis LUT:           int8 o int16
accumulatore:        int32
output:              float, int16 o int32 requantizzato
```

### Due versioni da testare

Versione più stabile:

```text
coefficients: int8
basis:        int16
accumulator:  int32
```

Versione più aggressiva e perfetta per `KAN.DOT4`:

```text
coefficients: int8
basis:        int8
accumulator:  int32
```

### Dove implementarla

Nel generatore Python:

```text
scripts/json_to_header.py
```

oppure in un nuovo script:

```text
scripts/json_to_quant_header.py
```

L’header dovrebbe contenere:

```c
static const uint32_t KAN_LAYER_COEFF4_Q8[...];
static const uint32_t KAN_BASIS_LUT_Q8[...];
static const float KAN_LAYER_SCALES[...];
```

### Metriche da misurare

Per ogni variante:

```text
MSE
MAE
MAX_ABS_ERROR
CHECKSUM
simInsts
cycles
IPC
L1D misses
L2 misses
```

### Perché è importante

La quantizzazione non è solo compressione. In questo progetto serve a trasformare la KAN in un workload compatibile con:

```text
RV32 packed int8
custom dot4
scratchpad compatto
meno traffico memoria
```

---

## 6. Implementare `dot4_scalar` come ponte verso la custom instruction

**Importanza:** alta  
**Difficoltà:** bassa  
**Dove inserirla:** nuovo `src/dot4.h`, oppure `src/bspline.c`.

### Perché serve

Prima di modificare gem5, conviene validare tutto in C puro.

Implementa:

```c
static inline int32_t dot4_scalar(uint32_t c, uint32_t b) {
    int32_t c0 = (int8_t)((c >> 0)  & 0xff);
    int32_t c1 = (int8_t)((c >> 8)  & 0xff);
    int32_t c2 = (int8_t)((c >> 16) & 0xff);
    int32_t c3 = (int8_t)((c >> 24) & 0xff);

    int32_t b0 = (int8_t)((b >> 0)  & 0xff);
    int32_t b1 = (int8_t)((b >> 8)  & 0xff);
    int32_t b2 = (int8_t)((b >> 16) & 0xff);
    int32_t b3 = (int8_t)((b >> 24) & 0xff);

    return c0*b0 + c1*b1 + c2*b2 + c3*b3;
}
```

### Kernel atteso

```c
acc += dot4_scalar(coeff4, basis4);
```

### Perché è importante

Questo step valida:

- packing dei coefficienti;
- packing delle basis;
- correttezza numerica;
- compatibilità con la LUT;
- futura semantica di `KAN.DOT4`.

---

## 7. Implementare custom instruction `KAN.DOT4`

**Importanza:** altissima  
**Difficoltà:** alta  
**Dove inserirla:** decoder RISC-V di gem5 + wrapper C inline assembly.

### Semantica

```asm
kan.dot4 rd, rs1, rs2
```

dove:

```text
rs1 = [c0 c1 c2 c3]
rs2 = [b0 b1 b2 b3]
rd  = c0*b0 + c1*b1 + c2*b2 + c3*b3
```

### Wrapper C

```c
static inline int32_t kan_dot4(uint32_t coeff4, uint32_t basis4)
{
    int32_t out;
    asm volatile (
        ".insn r CUSTOM_0, 0, 0x2a, %0, %1, %2"
        : "=r"(out)
        : "r"(coeff4), "r"(basis4)
    );
    return out;
}
```

### Kernel

```c
acc += kan_dot4(coeff4, basis4);
```

### Perché è importante

È il contributo hardware più forte:

```text
spline cubica
    ↓
4 basi attive
    ↓
4 coefficienti locali
    ↓
4 int8 dentro un registro RV32
    ↓
custom instruction KAN.DOT4
```

Questa istruzione è specifica per la struttura della KAN, non è una generica ottimizzazione da MLP.

---

## 8. Aggiungere scratchpad memory

**Importanza:** alta  
**Difficoltà:** alta  
**Dove inserirla:** gem5 config + codice C memory-mapped.

### Stato attuale

Oggi tutti i dati passano da:

```text
DDR3
  ↓
L2
  ↓
L1D
  ↓
CPU
```

Non esiste una memoria locale software-managed.

### Prima versione

Aggiungere una scratchpad memory-mapped, per esempio:

```text
0x90000000 - 0x90000fff
```

Dentro mettere:

```text
basis_lut
coeff_tile
input_tile
acc_tile
```

### Pipeline

```text
main memory
    ↓ copia tile
scratchpad
    ↓
kernel local4/dot4
    ↓
output in main memory
```

### Perché farla dopo LUT/packing

La scratchpad ha senso solo quando i dati sono già compatti e regolari.

Se la implementi prima, rischi di copiare tanti float poco riutilizzati e avere poco guadagno.

---

## 9. Scratchpad tiled + double buffering

**Importanza:** media-alta  
**Difficoltà:** alta  
**Dove inserirla:** gestione scratchpad e loop per tile.

### Idea

Dividere la scratchpad in due buffer:

```text
buffer A = tile corrente
buffer B = tile successivo
```

Pipeline:

```text
load tile A
compute tile A while loading tile B
compute tile B while loading tile C
...
```

### Perché è interessante

Riduce il tempo visibile di trasferimento dati e rende il progetto molto più hardware-oriented.

È una buona estensione dopo aver fatto scratchpad semplice.

---

## 10. Batching di più input

**Importanza:** media  
**Difficoltà:** media  
**Dove inserirla:** `src/main.c`, `kan_inference.c`, kernel ottimizzato.

### Stato attuale

Oggi il codice fa:

```c
for (int i = 0; i < n; ++i) {
    y_pred = kan_infer(x_i);
}
```

Ogni input viene trattato indipendentemente.

### Idea

Processare più input insieme:

```text
batch di 4, 8, 16 input
```

Così puoi:

- caricare coefficienti una volta;
- usarli per più sample;
- migliorare il riuso in cache/scratchpad;
- aumentare operational intensity.

### Perché non farlo subito

Prima conviene sistemare local4, LUT, quantizzazione e dot4. Il batching diventa molto più efficace dopo.

---

# Ordine finale consigliato

```text
1. Isolare meglio la misura della sola inferenza
2. Ottimizzare bspline_eval calcolando solo 4 basi attive
3. Cambiare layout dei coefficienti in coeff4 per intervallo
4. Aggiungere basis LUT float
5. Quantizzare coeff4 e basis4
6. Implementare dot4_scalar
7. Implementare KAN.DOT4 in gem5
8. Aggiungere scratchpad semplice
9. Aggiungere scratchpad tiled/double buffering
10. Aggiungere batching e prefetching
```

## Versione minima forte

```text
local4 B-spline
+ basis LUT
+ int8 packed
+ KAN.DOT4
```

## Versione completa quasi da paper

```text
local4 B-spline
+ basis LUT
+ quantizzazione int8
+ packed coeff layout
+ KAN.DOT4 custom instruction
+ scratchpad tiled
+ design-space exploration gem5
```

## Prossimo commit consigliato

```text
Implement local active-basis B-spline evaluation
```

Motivo: è il ponte naturale tra il fix già fatto, cioè ricorsivo → iterativo, e tutte le ottimizzazioni hardware successive.


---

# Appendice: ottimizzazioni concrete del codice C

Questa sezione aggiunge dettagli più pratici su come rendere più efficiente il codice attuale, oltre alla roadmap principale.

---

## A. Calcolare solo le basi attive locali

### Problema attuale

La versione iterativa attuale di `bspline_eval` è già migliore della vecchia ricorsiva, ma calcola ancora tutte le basi del grado finale e poi combina tutti i control points:

```c
for (int i = 0; i < num_control_points; ++i) {
    y += control_points[i] * prev[i];
}
```

Con `num_control_points = 15`, questo significa che per ogni edge vengono considerate 15 basi.

### Osservazione

Per una B-spline di grado `p`, se `x` cade nell'intervallo `[t_k, t_{k+1})`, sono non nulle al massimo `p + 1` basi.

Nel progetto:

```text
degree = 3
p + 1 = 4 basi attive
```

Quindi è inutile calcolare e sommare tutte le 15 basi.

### Nuovo obiettivo

Implementare una funzione:

```c
static inline float bspline_eval_local4(
    float x,
    const float *restrict knots,
    const float *restrict control_points
);
```

oppure una versione più generale:

```c
static inline float bspline_eval_local(
    float x,
    const float *restrict knots,
    const float *restrict control_points,
    int degree,
    int num_control_points,
    int num_knots
);
```

La versione ottimizzata deve fare:

```text
1. trovare l'intervallo k in cui cade x;
2. calcolare solo le 4 basi cubiche locali;
3. caricare solo i 4 control points corrispondenti;
4. restituire c0*b0 + c1*b1 + c2*b2 + c3*b3.
```

### Perché conviene

Riduci:

```text
- numero di basi calcolate;
- accessi a control_points;
- moltiplicazioni;
- somme;
- lavoro su prev[] e curr[];
- pressione su L1D;
- numero di istruzioni.
```

Inoltre prepari direttamente il codice a:

```text
basis4 + coeff4 → dot4
```

---

## B. Ricostruire solo il sotto-albero necessario

### Idea

La ricorrenza Cox-de Boor completa può essere vista come un albero. La versione ricorsiva calcolava tanti sottoalberi ripetuti. La versione iterativa attuale evita la ricorsione, ma costruisce ancora tutte le basi.

La prossima ottimizzazione è costruire solo il sottoalbero locale delle basi che possono essere non nulle.

Per `degree = 3`, se `x` cade in `[t_k, t_{k+1})`, le basi finali attive sono una finestra di 4 basi:

```text
B_{k-3,3}(x), B_{k-2,3}(x), B_{k-1,3}(x), B_{k,3}(x)
```

a seconda della convenzione degli indici e dei knot duplicati.

### Implementazione concettuale

```c
int k = find_knot_span(x, knots, num_knots);

float b[4];

compute_cubic_active_basis(
    x,
    knots,
    k,
    b
);

return control_points[k-3] * b[0]
     + control_points[k-2] * b[1]
     + control_points[k-1] * b[2]
     + control_points[k]   * b[3];
```

Servono controlli ai bordi perché vicino a `x_min` e `x_max` alcuni indici possono uscire dal range.

### Versione robusta

Per evitare bug ai bordi:

```text
1. trova k;
2. calcola start = k - degree;
3. clamp start a [0, num_control_points - 4];
4. calcola le 4 basi coerenti con quello start;
5. combina solo quei 4 coefficienti.
```

---

## C. Evitare ricerca lineare dell'intervallo quando possibile

### Stato base

Un modo semplice per trovare `k` è:

```c
for (int i = 0; i < num_knots - 1; ++i) {
    if (knots[i] <= x && x < knots[i + 1]) {
        return i;
    }
}
```

Per pochi knots va bene, ma è comunque un loop e contiene branch.

### Ottimizzazione per griglia uniforme

Nel tuo JSON i knots sembrano uniformemente spaziati, con estensione oltre `[0,1]`.

Se la griglia è uniforme, puoi calcolare l'intervallo con una formula:

```c
idx = (int)((x - knot_min) * inv_delta);
```

dove:

```c
inv_delta = 1.0f / delta;
```

Poi fai clamp:

```c
if (idx < 0) idx = 0;
if (idx > max_idx) idx = max_idx;
```

### Perché conviene

Riduce:

```text
- loop sui knots;
- branch;
- load ripetuti da KAN_LAYER_KNOTS;
- istruzioni.
```

È anche molto più compatibile con fixed-point e LUT.

---

## D. Precomputare denominatori e inverse denominator

Nella ricorrenza Cox-de Boor compaiono termini come:

```c
left_den  = knots[i + degree]     - knots[i];
right_den = knots[i + degree + 1] - knots[i + 1];
```

e poi divisioni:

```c
(x - knots[i]) / left_den
(knots[i + degree + 1] - x) / right_den
```

Le divisioni float sono costose.

### Ottimizzazione

Precomputare:

```c
inv_left_den
inv_right_den
```

nell'header generato, oppure in una struttura ausiliaria.

Allora a runtime fai:

```c
left = (x - knots[i]) * inv_left_den * prev[i];
```

invece di:

```c
left = ((x - knots[i]) / left_den) * prev[i];
```

### Dove inserirla

Nel generatore:

```text
scripts/json_to_header.py
```

creando array tipo:

```c
KAN_LAYER_INV_DENOMS[...]
```

oppure, se passi alla LUT, questa ottimizzazione diventa meno prioritaria perché le basis vengono precomputate.

---

## E. Usare `static inline` per funzioni piccole

### Funzioni candidate

```c
find_knot_span(...)
compute_cubic_active_basis(...)
dot4_scalar(...)
pack4_i8(...)
requantize(...)
```

### Esempio

```c
static inline int find_knot_span_uniform(float x)
{
    int idx = (int)((x - KNOT_MIN) * INV_DELTA);
    if (idx < 0) idx = 0;
    if (idx > MAX_IDX) idx = MAX_IDX;
    return idx;
}
```

### Perché conviene

Per funzioni molto piccole chiamate dentro loop interni, `inline` evita overhead di chiamata e permette al compilatore di ottimizzare meglio costanti, registri e loop.

Nota: `inline` non garantisce sempre l'inlining, ma con `-O2` o `-O3` spesso aiuta.

---

## F. Usare `restrict` sui puntatori

### Problema

Il compilatore C è conservativo: se vede due puntatori, deve assumere che possano puntare alla stessa memoria.

Esempio:

```c
float f(float *a, float *b)
```

Il compilatore non sa se `a` e `b` aliasano.

### Ottimizzazione

Usare:

```c
float bspline_eval_local4(
    float x,
    const float *restrict knots,
    const float *restrict control_points
);
```

Questo dice al compilatore:

```text
knots e control_points non si sovrappongono.
```

### Perché conviene

Può migliorare:

```text
- scheduling dei load;
- registri;
- vettorizzazione futura;
- eliminazione di reload inutili.
```

---

## G. Allineamento della memoria

### Stato attuale

Gli array `static const float` sono probabilmente già ragionevolmente allineati, ma non lo stai controllando esplicitamente.

### Ottimizzazione

Nel generatore header puoi emettere:

```c
#define KAN_ALIGN64 __attribute__((aligned(64)))

static const float KAN_ALIGN64 KAN_LAYER_CONTROL_POINTS[...] = {
    ...
};
```

Per packed int8:

```c
static const uint32_t KAN_ALIGN64 KAN_LAYER_COEFF4_Q8[...] = {
    ...
};
```

### Perché conviene

L'allineamento aiuta:

```text
- cache line access;
- load word allineati;
- futura SIMD/custom instruction;
- meno rischio di accessi misaligned;
- più prevedibilità nei benchmark.
```

---

## H. Ridurre array temporanei nello stack

### Stato attuale

`bspline_eval` usa:

```c
float prev[num_intervals];
float curr[num_intervals];
```

Questi array sono su stack.

### Ottimizzazione local4

Se calcoli solo 4 basi attive, non servono più array grandi.

Puoi usare piccoli array fissi:

```c
float b0[4];
float b1[4];
float b2[4];
float b3[4];
```

oppure direttamente variabili scalari.

Per spline cubiche, la versione più spinta può evitare quasi completamente array variabili.

### Perché conviene

Riduce:

```text
- accessi allo stack;
- load/store locali;
- pressione su L1D;
- overhead di inizializzazione.
```

---

## I. Specializzare per `degree = 3`

### Stato attuale

Il codice è generale:

```c
for (int current_degree = 1; current_degree <= degree; ++current_degree)
```

Ma nella repo i modelli usano sempre:

```text
degree = 3
```

### Ottimizzazione

Scrivere una versione specializzata:

```c
float bspline_eval_cubic_local4(...)
```

senza loop su `degree`.

### Perché conviene

Il compilatore può ottimizzare molto meglio:

```text
- loop unrolled;
- meno branch;
- meno indici dinamici;
- meno accessi a memoria;
- più registri.
```

Mantieni comunque la versione generale come fallback/debug.

---

## J. Separare versione reference e versione optimized

Non sostituire subito tutto.

Crea due funzioni:

```c
float bspline_eval_reference(...);
float bspline_eval_optimized(...);
```

Poi abilita con macro:

```c
#ifdef KAN_USE_OPT_BSPLINE
    spline = bspline_eval_optimized(...);
#else
    spline = bspline_eval_reference(...);
#endif
```

### Perché conviene

Ti permette di fare:

```text
- confronto numerico;
- ablation study;
- benchmark prima/dopo;
- debug più facile;
- paper più chiaro.
```

---

## K. Cose già presenti nel file TODO

Nel file TODO erano già presenti:

```text
- calcolo solo delle 4 basi attive;
- layout coeff4 per intervallo;
- LUT per basis spline;
- quantizzazione int8/int16;
- dot4_scalar;
- custom instruction KAN.DOT4;
- scratchpad;
- double buffering;
- batching.
```

Non erano ancora espanse in dettaglio queste micro-ottimizzazioni:

```text
- static inline;
- restrict;
- allineamento 64 byte;
- precomputazione inverse denominator;
- ricerca branchless/uniforme dell'intervallo;
- specializzazione degree=3;
- riduzione dello stack;
- separazione reference/optimized.
```

Questa appendice le aggiunge esplicitamente.
