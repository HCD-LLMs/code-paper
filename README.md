```markdown
# Prompt Engineering con Ollama + LangChain + MLflow

Questa guida mostra come configurare un ambiente locale per tracciare esperimenti con Ollama usando LangChain e MLflow, sia su Windows sia su macOS.

---

## Contenuti

1. [Preparazione dell’ambiente (venv)](#preparazione-dellambiente-venv)  
2. [Installazione delle dipendenze](#installazione-delle-dipendenze)  
3. [Avvio del server MLflow](#avvio-del-server-mlflow)  
4. [Avvio di Ollama (run model)](#avvio-di-ollama-run-model)  
5. [Esecuzione dello script Python](#esecuzione-dello-script-python)

---

## Preparazione dell’ambiente (venv)

### Windows

```powershell
cd C:\prompt_ollama
python -m venv .venv
.venv\Scripts\activate
```

### macOS

```bash
cd ~/prompt_ollama
python3 -m venv .venv
source .venv/bin/activate
```

---

## Installazione delle dipendenze

Con il virtual environment attivo, installa i pacchetti dal `requirements.txt`:

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```





## Avvio del server MLflow
In un altro terminale, al di fuori del virtual enviroment avviare il server locale di MLflow

### Windows

```powershell
mlflow server `
  --backend-store-uri file:./mlruns `
  --default-artifact-root file:./mlruns/artifacts `
  --host 127.0.0.1 --port 5000
```

### macOS

```bash
mlflow server \
  --backend-store-uri file:./mlruns \
  --default-artifact-root file:./mlruns/artifacts \
  --host 127.0.0.1 --port 5000
```

- **Dashboard**: apri il browser su [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Avvio di Ollama (run model)

Assicurati di aver installato Ollama:



```powershell

ollama run deepseek-r1:8b (o modello associato)
```



- Il modello verrà caricato e il server ascolterà su `http://localhost:11434`

---

## Esecuzione dello script Python

Con entrambi i servizi (MLflow e Ollama) in esecuzione e il venv attivo, lancia sul venv:

```bash
python mlflow_langchain_html.py
```

- Vedrai in console le risposte del modello  
- In MLflow compariranno le run con prompt, risposta, parametri e artefatti  

---

**Note finali**  
- Mantieni sempre il venv attivo quando lavori sul progetto.  
- Per aggiungere nuovi template, modifica `templates.py`.  
- Esegui il backup di `./mlruns` per salvare i log, rinomina la cartella nel seguente formtato `mlruns_nome_personale`.  
- Per aggiornare dipendenze, ricrea il venv partendo dal `requirements.txt`.  

Buon prompt engineering!