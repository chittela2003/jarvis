import ollama

history = []

while True:
    text = input("You: ")
    history.append({"role": "user", "content": text})
    reply = ollama.chat(model="llama3.2", messages=history)
    answer = reply["message"]["content"]
    history.append({"role": "assistant", "content": answer})
    print("Jarvis:", answer)