import os
import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware # <--- GARANTA QUE ESTE IMPORT ESTÁ AQUI
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (para uso local)
load_dotenv()

app = FastAPI()

# --- CONFIGURAÇÃO DE CORS (A PARTE MAIS IMPORTANTE) ---
# Lista de endereços (origens) que podem acessar sua API
origins = [
    "https://chic-sopapillas-c6ed6d.netlify.app", # <--- A URL EXATA DO SEU SITE NO NETLIFY
    "http://localhost:5500", # Para testes locais no futuro
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # <--- AQUI ESTÁ A PERMISSÃO
    allow_credentials=True,
    allow_methods=["*"], # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"], # Permite todos os cabeçalhos
)

# --- Modelos de Dados (Pydantic) ---
class RegisterData(BaseModel):
    name: str
    email: EmailStr
    password: str

class LinkData(BaseModel):
    name: str
    url: str

# --- Endpoints da API ---

@app.post("/auth/register")
def register_user(data: RegisterData):
    """Endpoint de registro de usuário (simulado)."""
    return {
        "token": "fake-jwt-token",
        "user": { "name": data.name, "email": data.email }
    }

@app.post("/links")
async def adicionar_link(data: LinkData):
    """Recebe um novo link e o salva no banco de dados."""
    DATABASE_URL = os.getenv('DATABASE_URL')
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute('INSERT INTO links (name, url) VALUES ($1, $2)', data.name, data.url)
        await conn.close()
        return {"status": "success", "message": "Link adicionado."}
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(status_code=400, detail="Esta URL já foi adicionada.")
    except Exception as e:
        print(f"Erro ao adicionar link: {e}")
        raise HTTPException(status_code=500, detail="Erro ao adicionar link.")

@app.get("/links")
async def buscar_links():
    """Busca e retorna todos os links salvos no banco de dados."""
    DATABASE_URL = os.getenv('DATABASE_URL')
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch('SELECT id, name, url, created_at FROM links ORDER BY created_at DESC')
        await conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Erro ao buscar links: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar links.")

@app.get("/imoveis")
async def buscar_imoveis():
    """Busca os dados REAIS dos imóveis do banco de dados."""
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="URL do banco de dados não configurada.")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch('SELECT * FROM imoveis')
        await conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"Erro ao buscar dados no banco: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao buscar dados dos imóveis.")