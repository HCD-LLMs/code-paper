from openai import OpenAI
import mlflow

# MLflow setup (locale)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Prompt_Comparison")
mlflow.openai.autolog()  # intercetta tutte le chiamate OpenAI

# Client Ollama/OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1/",  # notare la slash finale
    api_key="ollama"                         # richiesta ma ignorata da Ollama
)

# Esempio di chiamata
response = client.chat.completions.create(
    model="deepseek-r1:8b",
    messages=[
        {"role": "system", "content": "Rispondi in modo chiaro."},
        {"role": "user",   "content": "Cos'è un buco nero?"}
    ]
)
print(response.choices[0].message.content)
