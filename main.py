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

# (imports e código CORS no topo do arquivo)
# ...

# Modelo para receber os dados do link do front-end
class LinkData(BaseModel):
    name: str
    url: str

# --- INÍCIO DO CÓDIGO PARA ADICIONAR ---

@app.post("/links")
async def adicionar_link(data: LinkData):
    """
    Recebe um novo link do front-end e o salva
    na tabela 'links' do banco de dados.
    """
    DATABASE_URL = os.getenv('DATABASE_URL')
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # Insere o novo link na tabela
        await conn.execute(
            'INSERT INTO links (name, url) VALUES ($1, $2)',
            data.name, data.url
        )
        await conn.close()
        return {"status": "success", "message": "Link adicionado com sucesso."}
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(status_code=400, detail="Esta URL já foi adicionada.")
    except Exception as e:
        print(f"Erro ao adicionar link: {e}")
        raise HTTPException(status_code=500, detail="Erro ao adicionar link.")

@app.get("/links")
async def buscar_links():
    """
    Busca e retorna todos os links que estão salvos
    na tabela 'links' do banco de dados.
    """
    DATABASE_URL = os.getenv('DATABASE_URL')
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch('SELECT id, name, url, created_at FROM links ORDER BY created_at DESC')
        await conn.close()
        
        links = [dict(row) for row in rows]
        return links
    except Exception as e:
        print(f"Erro ao buscar links: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar links.")

# --- FIM DO CÓDIGO PARA ADICIONAR ---

# ... (seu endpoint /imoveis aqui) ...

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
