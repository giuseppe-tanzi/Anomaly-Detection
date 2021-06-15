# Anomaly Detector for Pedestrian Areas

## Indice

1. [Introduzione](#1-introduzione)
2. [Struttura Progetto](#2-struttura-progetto)
3. [Requisiti Per Eseguire Il Progetto](#3-requisiti-per-eseguire-il-progetto)
4. [Sviluppi Futuri](#4-sviluppi-futuri)

## 1. Introduzione

Il progetto è stato creato per sostenere l'esame di *Modelli e Metodi per la Sicurezza delle Applicazioni*.

## 2. Struttura Progetto

Il progetto è strutturato nel seguente modo:

```
|–– output
|    |–– Test001
|    |–– Test002
|    |      .
|    |      .
|    |      .
|    |–– Test035
|–– parameters
|    |–– autoencoder_ucsd_convLSTMAE.params
|    |--UCSD_Anomaly_Dataset
|–– .gitignore
|-- convLSTMAE.py
|-- Documentazione.pdf
|-- LICENSE
|-- main.py
|–– README.md
|–– utils.py
```

Nel seguito si dettagliano i ruoli dei diversi componenti:

- **output**: cartella in cui sono presenti i frame output dei 35 test con le anomalie evidenziate in rosso;
- **parameters**: cartella in cui è presente il file *.params* salvato dopo aver completato il train della rete neurale;
- **UCSD_Anomaly_Dataset**: dataset utilizzato per lo sviluppo del sistema
- **main.py**: file sorgente utilizzato come main del progetto;
- **convLSTMAE.py**: file sorgente utilizzato per definire la rete neurale e il corrispettivo train;
- **utils.py** file sorgente utilizzato per definire la creazione del dataloader e del plot dei frame con le anomalie
  evidenziate;
- **.gitignore**: file che specifica tutti i file che devono essere esclusi dal sistema di controllo versione;
- **Documentazione.pdf**: documentazione del caso di studio.

## 3. Requisiti Per Eseguire Il Progetto

Per eseguire il progetto è necessario installare i seguenti programmi:

- `Python 3.6.0`
- `Mxnet 1.6.0`
- `Mxnet-cu101 1.5.0`
- `Matplotlib 3.3.4`
- `Pillow 8.2.0`
- `Scipy 1.5.4`
- `Numpy 1.16.6`
- `Cuda 10.1`
- `cudnn for Cuda 10.1 - v.7.6.5.32`

Se non si dispone di una GPU, cambiare alla riga 13 del file `main.py`, scrivendo `ctx = mx.cpu()` per poter effettuare
il train con sola CPU.

## 4. Sviluppi Futuri

In futuro, il sistema sviluppato potrà essere utilizzato in larga scala da più comuni e metropoli per l'identificazione
di anomalie in zone pedonali ma anche in diversi contesti d'uso: un esempio consiste nel riallenare il software per le
anomalie nelle strade urbane e extra-urbane. Per far ciò sarà necessario utilizzare un dataset diverso da quello
utilizzato fino ad ora.
