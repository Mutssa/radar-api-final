import os
import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (para uso local)
load_dotenv()

app = FastAPI()

# --- Configuração de CORS ---
# No futuro, troque "*" pela URL do seu site no Netlify
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos de Dados (Pydantic) ---
class RegisterData(BaseModel):
    name: str
    email: EmailStr
    password: str

# --- Endpoints da API ---
@app.post("/auth/register")
def register_user(data: RegisterData):
    """Endpoint de registro de usuário (simulado)."""
    return {
        "token": "fake-jwt-token",
        "user": {
            "name": data.name,
            "email": data.email
        }
    }

@app.get("/imoveis")
async def buscar_imoveis():
    """
    Endpoint que busca os dados REAIS dos imóveis
    do seu banco de dados no Supabase.
    """
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="URL do banco de dados não configurada.")

    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch('SELECT * FROM imoveis')
        await conn.close()
        
        # Converte os resultados do banco para um formato JSON amigável
        imoveis = [dict(row) for row in rows]
        return imoveis

    except Exception as e:
        print(f"Erro ao buscar dados no banco: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao buscar dados dos imóveis.")
