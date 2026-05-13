# Modifica: knot uniformi compatti per inferenza KAN/gem5

Data: 2026-05-13

## Obiettivo

Ridurre la dimensione degli header C generati per l'inferenza KAN, evitando di salvare esplicitamente tutto il knot vector quando i knot sono uniformemente spaziati.

La matematica della spline non cambia. Cambia solo il modo in cui i knot vengono rappresentati nel codice C generato.

Prima:

```c
KAN_LAYER_KNOTS[layer][input][knot_index]
```

contieneva tutti i valori del knot vector per ogni layer e per ogni input.

Ora, se il knot vector e' uniforme, viene salvato solo:

```c
KAN_LAYER_KNOT_BASE[layer][input]
KAN_LAYER_KNOT_DELTA[layer][input]
```

e il knot viene ricostruito al volo come:

```c
t_i = knot_base + i * knot_delta
```

## Motivazione

Nei modelli KAN addestrati con griglia uniforme, per esempio con:

```text
grid = 5
degree = 3
num_control_points = 8
num_knots = 12
```

i knot sono equidistanti. Un esempio tipico e':

```text
[-2.2, -1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8, 2.2]
```

In questo caso salvare tutti i 12 valori per ogni input e layer e' ridondante, perche bastano il primo valore e il passo.

La modifica diventa particolarmente utile per il modello NASA, che ha molti input. In quel caso la tabella completa dei knot crescerebbe come:

```text
num_layers * max_input_dim * num_knots
```

mentre la versione compatta usa:

```text
num_layers * max_input_dim * 2
```

piu alcuni flag interi.

## File modificati

### `src_inference/scripts/json_to_header.py`

Il generatore dell'header ora controlla se ogni knot vector e' uniforme.

Sono state aggiunte queste funzioni:

```python
is_uniform_knot_vector(...)
build_knot_metadata(...)
```

Se tutti i knot vector attivi sono uniformi, il generatore:

- imposta `KAN_USE_IMPLICIT_UNIFORM_KNOTS` a `1`;
- genera `KAN_LAYER_KNOT_BASE`;
- genera `KAN_LAYER_KNOT_DELTA`;
- genera `KAN_LAYER_KNOTS_UNIFORM`;
- non genera piu la tabella completa `KAN_LAYER_KNOTS`.

Se invece in futuro un modello avesse knot non uniformi, il generatore mantiene la compatibilita:

- imposta `KAN_USE_IMPLICIT_UNIFORM_KNOTS` a `0`;
- genera comunque i metadati;
- mantiene anche la tabella completa `KAN_LAYER_KNOTS`;
- il codice C puo fare fallback alla valutazione classica.

Sono stati anche aggiunti gli alias:

```python
"nasa"
"nasa_kan"
```

che puntano all'export:

```text
artifacts/nasa_kan/nasa_kan_riscv_export.json
```

### `src_inference/include/bspline.h`

E' stata aggiunta la dichiarazione della nuova funzione:

```c
float bspline_eval_uniform(
    float x,
    float knot_base,
    float knot_delta,
    const float *control_points,
    int degree,
    int num_control_points,
    int num_knots
);
```

### `src_inference/src/bspline.c`

E' stata aggiunta la funzione:

```c
bspline_eval_uniform(...)
```

Questa funzione usa lo stesso algoritmo iterativo gia presente in `bspline_eval`, ma invece di leggere i knot da un array li calcola con:

```c
knot_base + knot_delta * index
```

E' stato aggiunto anche un controllo sul delta:

```c
if (fabsf(knot_delta) <= 1.0e-12f) {
    return 0.0f;
}
```

serve solo a evitare divisioni instabili in caso di export errati o degeneri.

### `src_inference/src/kan_inference.c`

La valutazione della spline ora sceglie automaticamente il percorso corretto:

1. Se `KAN_USE_IMPLICIT_UNIFORM_KNOTS == 1`, usa sempre `bspline_eval_uniform`.
2. Se esistono i metadati ma non tutti i knot sono uniformi, controlla il flag `KAN_LAYER_KNOTS_UNIFORM[layer][input]`.
3. Se il knot vector non e' uniforme, usa il vecchio `bspline_eval` con `KAN_LAYER_KNOTS`.
4. Se viene usato un vecchio header senza metadati, resta compatibile con il comportamento precedente.

Sono stati aggiunti default di compatibilita:

```c
#ifndef KAN_HAS_KNOT_METADATA
#define KAN_HAS_KNOT_METADATA 0
#endif

#ifndef KAN_USE_IMPLICIT_UNIFORM_KNOTS
#define KAN_USE_IMPLICIT_UNIFORM_KNOTS 0
#endif
```

## Comportamento dell'header generato

Per un modello con knot uniformi, l'header generato contiene:

```c
#define KAN_HAS_KNOT_METADATA 1
#define KAN_USE_IMPLICIT_UNIFORM_KNOTS 1

static const int KAN_LAYER_KNOTS_UNIFORM[...];
static const float KAN_LAYER_KNOT_BASE[...];
static const float KAN_LAYER_KNOT_DELTA[...];
```

e non contiene piu:

```c
static const float KAN_LAYER_KNOTS[...];
```

Questo riduce memoria statica e dimensione del file generato.

## Verifiche eseguite

Controllo sintattico del generatore:

```bash
python3 -m py_compile src_inference/scripts/json_to_header.py
```

Generazione header per modello piccolo:

```bash
cd src_inference
python3 scripts/json_to_header.py 1x4x1
```

Build host:

```bash
bash scripts/build_host.sh
```

Esecuzione host:

```bash
./build/host/kan_demo_host 16
```

Risultato osservato:

```text
MSE = 1.04740655e-06
MAE = 0.000752957509
MAX_ABS_ERROR = 0.00281482935
CHECKSUM = 0.00550558418
DONE
```

Build RISC-V:

```bash
bash scripts/build_riscv.sh
```

La build RISC-V e' passata.

Dopo le verifiche, `src_inference/include/kan_model.h` e' stato rimosso di nuovo per lasciare pulita la cartella `include`, dato che gli header modello sono generati.

## Nota sul modello NASA

Questa modifica prepara meglio il codice di inferenza per modelli piu grandi come NASA, perche evita di espandere inutilmente i knot uniformi nel file C.

Questo passaggio e' stato poi implementato nella sezione successiva. Per usare davvero il modello NASA in modo fedele serviva gestire:

- input vettoriale a 48 feature;
- normalizzazione delle feature coerente con training;
- branch `scale_base`;
- output scalato con `target_scale`;
- confronto numerico contro le predizioni PyTorch.

## Aggiornamento: inferenza NASA completa

Data: 2026-05-13

E' stata aggiunta una pipeline di inferenza C dedicata al modello NASA C-MAPSS, mantenendo separato il demo sintetico 1D.

La scelta e' stata:

- tenere il kernel KAN generale in `src_inference/src/kan_inference.c`;
- aggiungere `kan_infer_vector(...)` per input multidimensionale;
- mantenere `kan_infer(float x)` come wrapper per i vecchi demo 1D;
- aggiungere un main separato per NASA in `src_inference/src/nasa_main.c`;
- generare un header dati NASA temporaneo con `scripts/nasa_test_to_header.py`;
- non committare gli header generati `kan_model.h` e `nasa_test_data.h`.

### Perche un main separato

`src/main.c` resta pensato per il caso sintetico:

```text
x scalare -> target_function(x) -> metriche sintetiche
```

NASA invece richiede:

```text
48 feature -> KAN vettoriale -> output scalato in RUL -> metriche su Y_test
```

Mettere tutto nello stesso `main.c` avrebbe creato molti `#ifdef` e un flusso meno leggibile. Il kernel e' condiviso, ma il programma di benchmark e' separato.

### File aggiunti

#### `src_inference/src/nasa_main.c`

Esegue inferenza sul test set NASA generato in header.

Il flusso e':

```text
setup
warm-up
m5_reset_stats
loop kan_infer_vector sui campioni NASA
m5_dump_stats
calcolo metriche RUL
stampa risultati
```

La regione misurata resta pulita: dentro ci sono solo le chiamate a `kan_infer_vector(...)` e il salvataggio della predizione.

Il programma calcola:

- MSE;
- RMSE;
- MAE;
- massimo errore assoluto;
- checksum delle predizioni;
- differenza contro le predizioni PyTorch salvate in `artifacts/nasa_kan/test_predictions.csv`.

#### `src_inference/scripts/nasa_test_to_header.py`

Genera:

```text
src_inference/include/nasa_test_data.h
```

a partire da:

```text
datasets/NASA/processed/X_test.npy
datasets/NASA/processed/Y_test.npy
artifacts/nasa_kan/test_predictions.csv
```

Lo script replica il preprocessing usato nel training con `window_mode = "summary"`:

```text
mean
var
trend
```

Per ogni finestra temporale NASA produce quindi 48 feature:

```text
16 sensori * 3 statistiche = 48 feature
```

L'header generato contiene:

```c
NASA_TEST_FEATURES[N][48]
NASA_TEST_TARGETS[N]
NASA_TEST_REFERENCE_PREDICTIONS[N]
```

`NASA_TEST_REFERENCE_PREDICTIONS` serve solo per validare che il C stia riproducendo PyTorch.

#### `src_inference/scripts/build_nasa_host.sh`

Genera gli header e compila:

```text
build/host/nasa_kan_demo_host
```

Uso:

```bash
cd src_inference
bash scripts/build_nasa_host.sh nasa 0
./build/host/nasa_kan_demo_host
```

Il secondo argomento e' `max_samples`:

- `0` significa tutto il test set;
- un numero positivo limita il dataset generato, utile per prove rapide.

#### `src_inference/scripts/build_nasa_riscv.sh`

Genera gli header e compila:

```text
build/riscv/nasa_kan_demo_riscv
```

Uso:

```bash
cd src_inference
bash scripts/build_nasa_riscv.sh nasa 0
```

Il binario puo poi essere passato alle config gem5 gia esistenti con:

```bash
../gem5/build/RISCV/gem5.opt \
  --outdir=results/cache/nasa_kan \
  gem5-configs/riscv_cache.py \
  --binary build/riscv/nasa_kan_demo_riscv \
  --num-inputs 1024
```

### File modificati

#### `src_inference/include/kan_inference.h`

Sono ora esposte due funzioni:

```c
float kan_infer(float x);
float kan_infer_vector(const float *restrict input);
```

`kan_infer` resta compatibile con i vecchi demo scalari. `kan_infer_vector` e' la path reale usata da NASA.

#### `src_inference/src/kan_inference.c`

La forward pass ora supporta:

- input vettoriale;
- layout flat per edge;
- ramo spline;
- ramo base PyKAN:

```c
scale_base * SiLU(x)
```

La formula per edge e':

```text
mask * (scale_base * SiLU(x) + scale_sp * spline(x))
```

che corrisponde alla forward di `KANLayer.py`:

```python
y = scale_base * base_fun(x) + scale_sp * spline(x)
y = mask * y
y = torch.sum(y, dim=1)
```

#### `src_inference/scripts/json_to_header.py`

L'header modello non usa piu tensori pieni paddati per input/output massimi.

Prima:

```c
KAN_LAYER_CONTROL_POINTS
    [KAN_NUM_LAYERS]
    [KAN_MAX_INPUT_DIM]
    [KAN_MAX_OUTPUT_DIM]
    [KAN_NUM_CONTROL_POINTS]
```

Ora:

```c
KAN_EDGE_CONTROL_POINTS
    [KAN_NUM_EDGES]
    [KAN_NUM_CONTROL_POINTS]
```

e per layer:

```c
KAN_LAYER_EDGE_OFFSETS[layer]
```

L'indice edge e':

```c
edge = KAN_LAYER_EDGE_OFFSETS[layer] + input_index * output_dim + output_index
```

Per NASA:

```text
layer 0: 48 * 16 = 768 edge
layer 1: 16 * 8  = 128 edge
layer 2: 8 * 1   = 8 edge
totale: 904 edge
```

Questo evita il padding della vecchia forma:

```text
3 layer * 48 max input * 16 max output = 2304 edge teorici
```

Quindi i coefficienti spline passano da:

```text
2304 * 8 = 18432 float
```

a:

```text
904 * 8 = 7232 float
```

Per NASA le mask sono tutte 1, quindi il generatore imposta:

```c
#define KAN_HAS_EDGE_MASKS 0
```

e non emette l'array delle mask.

#### `src_inference/include/bspline.h`

E' stato aggiunto un fast path `static inline` per le spline cubiche uniformi:

```c
BsplineCubicLocalBasis
bspline_make_uniform_cubic_basis(...)
bspline_dot_uniform_cubic_basis(...)
bspline_eval_uniform_cubic_local(...)
```

Questa e' l'applicazione pratica del punto 2 del TODO: per grado 3 non si calcolano tutte le basi, ma solo le 4 basi locali attive.

### Economia delle basi

Per una spline cubica:

```text
degree = 3
basis attive = degree + 1 = 4
```

La path ottimizzata fa:

```text
1. trova lo span locale del knot vector uniforme;
2. calcola le 4 basi cubiche locali;
3. riusa quelle 4 basi per tutti gli output collegati allo stesso input;
4. per ogni edge fa solo il dot con i 4 control point attivi.
```

Questo e' piu economico della vecchia valutazione, che calcolava il vettore completo delle basi e poi sommava tutti i control point.

Nel caso NASA, il guadagno e' importante soprattutto nel primo layer:

```text
48 input
16 output
768 edge
```

Le basi dipendono da `input_index`, non da `output_index`. Quindi per il primo layer si calcolano 48 quartine di basi, non 768.

### Verifiche eseguite

Compilazione Python:

```bash
python3 -m py_compile \
  src_inference/scripts/json_to_header.py \
  src_inference/scripts/nasa_test_to_header.py
```

Verifica demo 1D:

```bash
cd src_inference
python3 scripts/json_to_header.py 1x4x1
bash scripts/build_host.sh
./build/host/kan_demo_host 16
```

Risultato:

```text
MSE = 1.04718287e-06
MAE = 0.000752866588
MAX_ABS_ERROR = 0.00281473063
CHECKSUM = 0.00550231151
DONE
```

Verifica NASA su tutto il test set:

```bash
cd src_inference
bash scripts/build_nasa_host.sh nasa 0
./build/host/nasa_kan_demo_host
```

Risultato:

```text
N = 10196
MSE = 191.179346
RMSE = 13.8267619
MAE = 9.61517945
MAX_ABS_ERROR = 57.5619049
MAE_VS_REFERENCE = 0.000119755108
MAX_ABS_VS_REFERENCE = 0.00048828125
DONE
```

La MAE PyTorch salvata negli artifact era:

```text
MAE = 9.61519718170166
```

Quindi la path C riproduce il modello PyTorch con errore numerico molto piccolo.

Build RISC-V NASA:

```bash
cd src_inference
bash scripts/build_nasa_riscv.sh nasa 0
```

Build RISC-V demo 1D:

```bash
cd src_inference
python3 scripts/json_to_header.py 1x4x1
bash scripts/build_riscv.sh
```

Entrambe le build sono passate.

Dopo le verifiche sono stati rimossi di nuovo:

```text
src_inference/include/kan_model.h
src_inference/include/nasa_test_data.h
```

perche sono header generati.
