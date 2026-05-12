# Pipeline di lettura di un singolo dato nella memoria

Questo documento spiega dove vengono caricati i dati del modello KAN e cosa succede quando un singolo coefficiente viene letto, assumendo che all'inizio le cache L1 e L2 siano vuote.

---

# 1. Punto fondamentale: l'header non viene caricato a runtime

`include/kan_model.h` non è un file letto durante l'esecuzione.

Non succede questo:

```text
apro kan_model.h
leggo i coefficienti
li carico in memoria
```

Succede invece questo:

```text
kan_model.h
    ↓ compilazione
build/riscv/kan_demo_riscv
    ↓ gem5 carica il binario ELF
memoria simulata DDR3
    ↓ load RISC-V quando serve un coefficiente
L2
    ↓
L1D
    ↓
registro CPU
```

Quindi l'header viene trasformato in dati binari dentro l'eseguibile.

---

# 2. Dove finiscono gli array dell'header

Nel file `kan_model.h` generato ci sono array come:

```c
static const float KAN_LAYER_CONTROL_POINTS[...];
static const float KAN_LAYER_KNOTS[...];
static const float KAN_LAYER_SCALE_SP[...];
static const float KAN_LAYER_MASK[...];
```

Essendo `static const float`, il compilatore li mette tipicamente nella sezione read-only del binario ELF:

```text
.rodata
```

Il binario RISC-V contiene quindi:

```text
build/riscv/kan_demo_riscv

.text      → istruzioni macchina
.rodata    → costanti, stringhe, coefficienti KAN, knots, mask
.data      → dati globali inizializzati modificabili
.bss       → dati globali inizializzati a zero
stack      → variabili locali runtime
heap       → malloc, se usato
```

Nel tuo caso:

```text
coefficienti KAN
knots
scale_sp
mask
stringhe printf
```

stanno nella sezione `.rodata`.

---

# 3. Cosa fa gem5 all'avvio

La configurazione gem5 crea:

```text
RISC-V SE mode
Timing CPU
1 core
L1I
L1D
L2 privata
DDR3 simulata
```

In SE mode gem5 non boota un sistema operativo completo. Carica il binario ELF nello spazio di memoria del processo simulato.

All'inizio della simulazione puoi immaginare:

```text
DDR3 simulata:
    contiene .text del programma
    contiene .rodata del programma
    contiene .data
    contiene .bss
    contiene stack iniziale

L1I:
    vuota

L1D:
    vuota

L2:
    vuota
```

Punto chiave:

```text
gem5 carica il programma nella memoria simulata,
ma non pre-riempie le cache.
```

Le cache si riempiono solo quando la CPU esegue fetch di istruzioni o load/store di dati.

---

# 4. Differenza tra istruzioni e dati

Nel processore simulato hai due flussi principali:

```text
istruzioni del programma → L1I
dati del modello         → L1D
stack/current/next       → L1D
```

Esempio:

```text
src/bspline.c compilato      → .text   → L1I
KAN_LAYER_CONTROL_POINTS     → .rodata → L1D
prev/curr/current/next       → stack   → L1D
```

---

# 5. Percorso di un singolo coefficiente

Prendiamo questo dato:

```c
KAN_LAYER_CONTROL_POINTS[layer][input_index][output_index][cp_index]
```

È un `float`, quindi occupa 4 byte.

In memoria C row-major, il suo indirizzo è concettualmente:

```c
addr =
    base(KAN_LAYER_CONTROL_POINTS)
    + (((layer * KAN_MAX_INPUT_DIM + input_index)
    * KAN_MAX_OUTPUT_DIM + output_index)
    * KAN_NUM_CONTROL_POINTS + cp_index) * sizeof(float);
```

Quando `kan_infer` chiama `bspline_eval`, passa un puntatore al vettore dei control points:

```c
KAN_LAYER_CONTROL_POINTS[layer][input_index][output_index]
```

Dentro `bspline_eval`, il dato viene letto quando si fa:

```c
y += control_points[i] * prev[i];
```

Quindi `control_points[i]` punta dentro `.rodata`.

---

# 6. Caso iniziale: L1 e L2 completamente vuote

Supponiamo che la CPU debba leggere per la prima volta:

```c
control_points[0]
```

La pipeline è:

```text
1. CPU calcola l'indirizzo di control_points[0]

2. CPU emette una load

3. L1D controlla se la cache line è presente

4. L1D miss, perché la cache è vuota

5. La richiesta va alla L2

6. L2 controlla se la cache line è presente

7. L2 miss, perché anche L2 è vuota

8. La richiesta va al memory controller DDR3 simulato

9. La DDR3 restituisce non solo il float, ma una cache line intera

10. La cache line viene inserita in L2

11. La cache line viene inserita in L1D

12. Il float richiesto viene mandato alla CPU

13. La CPU usa il dato nella moltiplicazione
```

Quindi il dato fa:

```text
DDR3 simulata
    ↓
L2 cache
    ↓
L1D cache
    ↓
registro CPU/FPU
    ↓
moltiplicazione
```

---

# 7. Non viene caricato un singolo float

Quando fai il primo accesso a un coefficiente, la memoria non porta in cache solo 4 byte.

Porta una cache line intera.

Se la cache line è da 64 byte:

```text
64 byte / 4 byte per float = 16 float
```

Quindi il primo accesso a:

```c
control_points[0]
```

può caricare nella stessa linea anche:

```c
control_points[1]
control_points[2]
...
control_points[15]
```

se sono contigui e allineati nella stessa cache line.

Perciò il primo accesso costa molto:

```text
L1D miss + L2 miss + DDR3 access
```

ma gli accessi immediatamente successivi a coefficienti vicini possono essere:

```text
L1D hit
```

---

# 8. La cache non carica tutto `kan_model.h`

Anche se `kan_model.h` contiene molti array, la cache non carica tutto l'header.

Carica solo le cache line effettivamente toccate.

Esempio:

```text
KAN_LAYER_CONTROL_POINTS = molti KB
```

ma il primo accesso porta magari solo:

```text
64 byte
```

cioè la cache line che contiene il coefficiente richiesto.

Man mano che il codice scorre i dati, vengono caricate altre cache line.

---

# 9. Pipeline concreta per un input `x`

Supponiamo:

```text
x = 0.37
```

Il programma principale fa:

```c
y_pred = kan_infer(x);
```

Dentro `kan_infer`:

```c
current[0] = x;
```

`current` è un array locale:

```c
float current[KAN_MAX_LAYER_WIDTH];
float next[KAN_MAX_LAYER_WIDTH];
```

Quindi sta sullo stack.

Il primo accesso allo stack può causare un miss in L1D, poi la cache line dello stack viene caricata.

---

# 10. Primo layer della KAN

Nel primo layer, `kan_infer` legge:

```c
KAN_LAYER_INPUT_DIMS[layer]
KAN_LAYER_OUTPUT_DIMS[layer]
```

Questi sono dati read-only in `.rodata`.

Con cache fredda:

```text
load dimensione layer
    ↓
L1D miss
    ↓
L2 miss
    ↓
DDR3
    ↓
L2 fill
    ↓
L1D fill
    ↓
dato alla CPU
```

Poi legge:

```c
KAN_LAYER_MASK[layer][input_index][output_index]
```

Anche questo è in `.rodata`.

Poi, se la mask è diversa da zero, chiama:

```c
bspline_eval(
    current[input_index],
    KAN_LAYER_KNOTS[layer][input_index],
    KAN_LAYER_CONTROL_POINTS[layer][input_index][output_index],
    ...
);
```

---

# 11. Dentro `bspline_eval`

La funzione attuale crea:

```c
float prev[num_intervals];
float curr[num_intervals];
```

Questi sono array locali su stack.

Quindi dentro `bspline_eval` hai due tipi di dati:

```text
1. Dati read-only del modello:
   - knots
   - control_points
   - scale_sp
   - mask
   - layer dims
   → stanno in .rodata

2. Dati temporanei runtime:
   - current[]
   - next[]
   - prev[]
   - curr[]
   - sum
   - y
   → stanno in registri e stack
```

La pipeline interna è:

```text
x
    ↓
lettura knots[i], knots[i+1]
    ↓
calcolo basi grado 0
    ↓
scrittura prev[]
    ↓
per ogni grado:
        lettura knots
        lettura prev[]
        calcolo left/right
        scrittura curr[]
        copia curr[] → prev[]
    ↓
lettura control_points[i]
    ↓
lettura prev[i]
    ↓
moltiplicazione
    ↓
accumulo in y
```

---

# 12. Percorso dei `knots`

Quando il codice legge:

```c
knots[i]
```

anche `knots` punta dentro `.rodata`.

Con cache fredda:

```text
knots[i]
    ↓
L1D miss
    ↓
L2 miss
    ↓
DDR3
    ↓
cache line con più knots
    ↓
L2
    ↓
L1D
    ↓
registro
```

Dato che i knots sono contigui, dopo il primo miss molti accessi successivi sono hit.

---

# 13. Percorso di `prev[]` e `curr[]`

`prev[]` e `curr[]` sono locali a `bspline_eval`.

Quindi stanno nello stack.

Primo accesso:

```text
stack address
    ↓
L1D miss possibile
    ↓
L2 miss possibile
    ↓
DDR3
    ↓
cache line dello stack in L2 e L1D
```

Poi gli accessi successivi a `prev[]` e `curr[]` sono probabilmente L1D hit perché gli array sono piccoli.

---

# 14. Percorso di `scale_sp` e `mask`

Dopo che `bspline_eval` ritorna, `kan_infer` fa:

```c
edge_value = KAN_LAYER_SCALE_SP[layer][input_index][output_index] * spline;
sum += mask * edge_value;
```

`scale_sp` e `mask` sono in `.rodata`.

Se non sono ancora in cache:

```text
L1D miss
    ↓
L2 miss
    ↓
DDR3
    ↓
L2 fill
    ↓
L1D fill
    ↓
registro
```

Se si trovano nella stessa cache line di dati già letti o sono già stati usati, possono essere L1D hit.

---

# 15. Pipeline completa di un edge

Per un singolo edge attivo:

```text
1. Leggi current[input_index]
   - stack o registro
   - L1D se serve

2. Leggi mask[layer][input][output]
   - .rodata
   - L1D/L2/DDR3 se miss

3. Leggi knots[layer][input]
   - .rodata
   - L1D/L2/DDR3 se miss

4. Leggi control_points[layer][input][output]
   - .rodata
   - L1D/L2/DDR3 se miss

5. Alloca/usa prev[] e curr[]
   - stack
   - L1D/L2/DDR3 se miss

6. Calcola spline con Cox-de Boor iterativo

7. Leggi scale_sp[layer][input][output]
   - .rodata

8. Calcola edge_value

9. Accumula sum

10. Scrivi next[output_index]
    - stack
```

---

# 16. Pipeline completa di un dato del modello

Per un coefficiente spline:

```text
JSON export
    ↓ prima della compilazione
scripts/json_to_header.py
    ↓
include/kan_model.h
    ↓ compilazione C
.rodata dentro build/riscv/kan_demo_riscv
    ↓ gem5 loader
DDR3 simulata
    ↓ primo load
L1D miss
    ↓
L2 miss
    ↓
DDR3 access
    ↓
cache line caricata
    ↓
L2 fill
    ↓
L1D fill
    ↓
registro CPU/FPU
    ↓
operazione floating point
```

Questa è la pipeline reale del dato.

---

# 17. Pipeline completa di un'istruzione

Per confronto, una istruzione di `bspline_eval` fa:

```text
.text del binario
    ↓
DDR3 simulata
    ↓ instruction fetch
L1I miss
    ↓
L2 miss
    ↓
DDR3 access
    ↓
L2 fill
    ↓
L1I fill
    ↓
decode/execute
```

Quindi:

```text
istruzioni → L1I
dati       → L1D
entrambi possono passare da L2 e DDR3
```

---

# 18. Cosa cambia con local4

Oggi:

```text
per ogni edge:
    leggo molti knots
    calcolo tutte le basi
    leggo 15 control_points
    faccio 15 moltiplicazioni
```

Dopo local4:

```text
per ogni edge:
    trovo intervallo k
    calcolo 4 basi attive
    leggo 4 control_points
    faccio 4 moltiplicazioni
```

Questo riduce:

```text
- load da .rodata
- moltiplicazioni
- somme
- loop interni
- pressione sulla cache
```

---

# 19. Cosa cambia con LUT + packed int8

Dopo LUT e quantizzazione:

```text
per ogni edge:
    idx = intervallo in cui cade x
    frac = posizione locale dentro l'intervallo
    basis4 = basis_lut[frac]
    coeff4 = coeff_packed[edge][idx]
    acc += dot4(coeff4, basis4)
```

La pipeline di un coefficiente diventa:

```text
coeff4 packed in .rodata
    ↓
load 32 bit
    ↓
L1D/L2/DDR3 se miss
    ↓
registro RV32
    ↓
dot4
```

Prima:

```text
15 float = 60 byte per edge
```

Dopo:

```text
4 int8 packed = 4 byte per valutazione locale
```

Quindi una singola cache line contiene molti più dati utili.

---

# 20. Cosa cambia con scratchpad

Oggi:

```text
.rodata in DDR3
    ↓
L2
    ↓
L1D
    ↓
CPU
```

Con scratchpad:

```text
.rodata in DDR3
    ↓ copia esplicita tile
scratchpad
    ↓
CPU/custom unit
```

La scratchpad non è una cache automatica. È una memoria locale software-managed.

Esempio:

```text
1. Copia coeff_tile dalla memoria normale alla scratchpad
2. Copia basis_lut o tienila residente
3. Esegui dot4 leggendo dalla scratchpad
4. Scrivi output
```

Questo serve a evitare miss e traffico irregolare nella gerarchia cache.

---

# 21. Riassunto finale

Il dato non nasce nella cache.

Nasce nel JSON, viene trasformato in header, viene compilato dentro il binario, viene caricato nella DRAM simulata da gem5, e solo quando la CPU lo richiede entra nelle cache.

Pipeline finale:

```text
JSON
  ↓
kan_model.h
  ↓
ELF RISC-V
  ↓
.rodata
  ↓
DDR3 simulata
  ↓
L2 cache
  ↓
L1D cache
  ↓
registro
  ↓
FPU/ALU
```

Con cache fredda, il primo accesso a un dato causa:

```text
L1D miss
L2 miss
DDR3 access
cache line fill
```

Gli accessi successivi a dati vicini possono essere molto più veloci grazie alla locality.
