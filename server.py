import hashlib
import json
import mimetypes
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from motor import buscar_y_filtrar, cargar_lista_blanca, extraer_dominio

DB_PATH = "fidesearch.db"
estado_app = {}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                perfil TEXT DEFAULT 'general',
                temas TEXT DEFAULT '[]',
                umbral_defecto INTEGER DEFAULT 55,
                entrevista_completada INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sesiones (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lista_personal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dominio TEXT NOT NULL,
                nota TEXT DEFAULT '',
                UNIQUE(user_id, dominio)
            )
            """
        )
        conn.commit()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return pwd_hash, salt


def obtener_usuario_por_token(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return None
    with get_db() as conn:
        fila = conn.execute(
            """
            SELECT u.id, u.username, u.perfil, u.temas, u.umbral_defecto, u.entrevista_completada
            FROM sesiones s
            JOIN usuarios u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if not fila:
            return None
        dominios = conn.execute(
            "SELECT dominio, nota FROM lista_personal WHERE user_id = ? ORDER BY id DESC",
            (fila["id"],),
        ).fetchall()
        return {
            "id": fila["id"],
            "username": fila["username"],
            "perfil": fila["perfil"],
            "temas": json.loads(fila["temas"] or "[]"),
            "umbral_defecto": fila["umbral_defecto"],
            "entrevista_completada": bool(fila["entrevista_completada"]),
            "lista_personal": [{"dominio": d["dominio"], "nota": d["nota"]} for d in dominios],
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    estado_app["modelo"] = joblib.load("modelo_fiabilidad.joblib")
    estado_app["lista_blanca"] = cargar_lista_blanca("fuentes_verificadas.csv")
    yield
    estado_app.clear()


app = FastAPI(title="FideSearch API", lifespan=lifespan)

mimetypes.add_type("image/svg+xml", ".svg")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class AuthPayload(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=4, max_length=100)


class PerfilPayload(BaseModel):
    perfil: str = Field(..., max_length=30)
    temas: list[str] = Field(default_factory=list)
    umbral_defecto: int = Field(55, ge=45, le=85)


class DominioPayload(BaseModel):
    dominio: str = Field(..., min_length=3, max_length=120)
    nota: str = Field("", max_length=80)


@app.get("/")
def servir_inicio():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/fuentes-globales")
def api_fuentes_globales():
    fuentes = [
        {"dominio": dom, "categoria": cat}
        for dom, cat in sorted(estado_app["lista_blanca"].items())
    ]
    return {"total": len(fuentes), "fuentes": fuentes}


@app.post("/api/auth/register")
def api_register(datos: AuthPayload):
    username = datos.username.strip().lower()
    if not username.isalnum():
        raise HTTPException(status_code=400, detail="El usuario solo puede tener letras y números.")

    pwd_hash, salt = hash_password(datos.password)
    token = secrets.token_urlsafe(32)
    try:
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO usuarios (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, pwd_hash, salt),
            )
            user_id = cur.lastrowid
            conn.execute("INSERT INTO sesiones (token, user_id) VALUES (?, ?)", (token, user_id))
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Ese nombre de usuario ya existe.")

    return {
        "token": token,
        "usuario": obtener_usuario_por_token(f"Bearer {token}"),
    }


@app.post("/api/auth/login")
def api_login(datos: AuthPayload):
    username = datos.username.strip().lower()
    with get_db() as conn:
        fila = conn.execute(
            "SELECT id, password_hash, salt FROM usuarios WHERE username = ?",
            (username,),
        ).fetchone()
        if not fila:
            raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos.")

        intento_hash, _ = hash_password(datos.password, fila["salt"])
        if not secrets.compare_digest(intento_hash, fila["password_hash"]):
            raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos.")

        token = secrets.token_urlsafe(32)
        conn.execute("INSERT INTO sesiones (token, user_id) VALUES (?, ?)", (token, fila["id"]))
        conn.commit()

    return {
        "token": token,
        "usuario": obtener_usuario_por_token(f"Bearer {token}"),
    }


@app.post("/api/auth/logout")
def api_logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        with get_db() as conn:
            conn.execute("DELETE FROM sesiones WHERE token = ?", (token,))
            conn.commit()
    return {"ok": True}


@app.get("/api/auth/me")
def api_me(authorization: str | None = Header(default=None)):
    usuario = obtener_usuario_por_token(authorization)
    if not usuario:
        raise HTTPException(status_code=401, detail="Sesión no válida o caducada.")
    return {"usuario": usuario}


@app.post("/api/perfil")
def api_guardar_perfil(datos: PerfilPayload, authorization: str | None = Header(default=None)):
    usuario = obtener_usuario_por_token(authorization)
    if not usuario:
        raise HTTPException(status_code=401, detail="Inicia sesión para guardar tu perfil.")

    with get_db() as conn:
        conn.execute(
            """
            UPDATE usuarios
            SET perfil = ?, temas = ?, umbral_defecto = ?, entrevista_completada = 1
            WHERE id = ?
            """,
            (datos.perfil, json.dumps(datos.temas), datos.umbral_defecto, usuario["id"]),
        )
        conn.commit()

    return {"usuario": obtener_usuario_por_token(authorization)}


@app.post("/api/mi-lista")
def api_anadir_dominio(datos: DominioPayload, authorization: str | None = Header(default=None)):
    usuario = obtener_usuario_por_token(authorization)
    if not usuario:
        raise HTTPException(status_code=401, detail="Inicia sesión para editar tu lista blanca.")

    dominio_limpio = extraer_dominio(datos.dominio)
    if "." not in dominio_limpio:
        raise HTTPException(status_code=400, detail="Introduce un dominio válido (ej. xataka.com o mit.edu).")

    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO lista_personal (user_id, dominio, nota) VALUES (?, ?, ?)",
            (usuario["id"], dominio_limpio, datos.nota.strip()),
        )
        conn.commit()

    return {"usuario": obtener_usuario_por_token(authorization)}


@app.delete("/api/mi-lista/{dominio}")
def api_borrar_dominio(dominio: str, authorization: str | None = Header(default=None)):
    usuario = obtener_usuario_por_token(authorization)
    if not usuario:
        raise HTTPException(status_code=401, detail="Inicia sesión para editar tu lista blanca.")

    with get_db() as conn:
        conn.execute(
            "DELETE FROM lista_personal WHERE user_id = ? AND dominio = ?",
            (usuario["id"], dominio.strip().lower()),
        )
        conn.commit()

    return {"usuario": obtener_usuario_por_token(authorization)}


@app.get("/api/buscar")
def api_buscar(
    q: str = Query(..., min_length=1, max_length=200),
    umbral: int = Query(55, ge=40, le=90),
    num: int = Query(10, ge=5, le=15),
    region: str = Query("wt-wt"),
    idioma: str = Query("es"),
    authorization: str | None = Header(default=None),
):
    consulta = q.strip()
    if not consulta:
        raise HTTPException(status_code=400, detail="La búsqueda no puede estar vacía.")

    usuario = obtener_usuario_por_token(authorization)
    lista_personal = [d["dominio"] for d in usuario["lista_personal"]] if usuario else None
    perfil = usuario["perfil"] if usuario else "general"

    inicio = time.time()
    try:
        res = buscar_y_filtrar(
            consulta=consulta,
            modelo=estado_app["modelo"],
            lista_blanca=estado_app["lista_blanca"],
            umbral=umbral / 100.0,
            num_resultados=num,
            region=region,
            idioma=idioma,
            lista_personal=lista_personal,
            perfil=perfil,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar internet: {e}")

    return {
        "consulta": consulta,
        "tiempo": round(time.time() - inicio, 2),
        "aprobados": res["aprobados"],
        "ocultos": res["ocultos"],
        "sintesis": res["sintesis"],
    }