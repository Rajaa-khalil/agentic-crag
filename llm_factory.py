import os
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(provider: str = "fallback", model_name: str = "qwen2.5:3b", temperature: float = 0.0) -> BaseChatModel:
    if provider == "ollama":
        print(f"🤖 Initializing Local Ollama Model ({model_name})...")
        return ChatOllama(model=model_name, temperature=temperature, base_url="http://localhost:11434")
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set!")
        return ChatGroq(model_name=model_name, temperature=temperature, api_key=api_key)
    elif provider == "fallback":
        try:
            llm = ChatOllama(model="qwen2.5:3b", temperature=temperature)
            llm.invoke("ping")
            print("✅ Using Local Ollama Engine.")
            return llm
        except Exception as e:
            print(f"⚠️ Local Ollama not responding ({e}). Switching to Groq Cloud...")
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")

if __name__ == "__main__":
    llm = get_llm("fallback")
    print(llm.invoke("Confirm you are ready to build the RAG agent.").content)