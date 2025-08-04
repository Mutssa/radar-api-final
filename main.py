from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Permitir requisições de qualquer origem (ajuste depois em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ['http://localhost:8080'] se você usar servidor local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo dos dados enviados no cadastro
class RegisterData(BaseModel):
    name: str
    email: EmailStr
    password: str

@app.post("/auth/register")
def register_user(data: RegisterData):
    # Aqui você pode salvar no banco real. Neste exemplo, só devolve os dados simulados.
    return {
        "token": "fake-jwt-token",
        "user": {
            "name": data.name,
            "email": data.email
        }
    }
