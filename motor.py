import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.parse import urlparse

import joblib
import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from deep_translator import GoogleTranslator


PATRONES_SENSACIONALISTAS = (
    "no quieren que sepas",
    "secreto oculto",
    "cura milagrosa",
    "cura definitiva",
    "100% garantizado",
    "comparte antes de que lo borren",
    "lo que nadie te cuenta",
    "médicos ocultaron",
    "medicos ocultaron",
    "what doctors don't want you to know",
    "miracle cure",
    "shocking truth",
    "they don't want you to know",
)

DOMINIOS_REFERENCIA_CITAS = (
    ".gob.", ".gov", ".edu", "doi.org", "who.int", "nih.gov",
    "nature.com", "science.org", "wikipedia.org", "csic.es", "europa.eu",
)

# Caché ligera en RAM (máx. 15 min) para acelerar búsquedas repetidas o cambios de umbral
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 900  # 15 minutos
_CACHE_BUSQUEDAS: dict[tuple, tuple[float, list[dict]]] = {}
_CACHE_WEBS: dict[str, tuple[float, dict]] = {}


def limpiar_texto(texto: str) -> str:
    texto = str(texto)
    texto = re.sub(r"\b(NUMBER|URL|EMAIL|PHONE|DOL|EUR)\b", " ", texto)
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def cargar_lista_blanca(ruta_csv: str = "fuentes_verificadas.csv") -> dict:
    df = pd.read_csv(ruta_csv)
    lista = {}
    for _, fila in df.iterrows():
        dom = str(fila["dominio"]).strip().lower().removeprefix("www.")
        cat = str(fila["categoria"]).strip()
        lista[dom] = cat
    return lista


def extraer_dominio(url: str) -> str:
    try:
        cand = url.strip()
        if not cand.startswith(("http://", "https://")):
            cand = "https://" + cand
        netloc = urlparse(cand).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""


def comprobar_capa_1(
    dominio: str,
    lista_blanca: dict,
    lista_personal: list[str] | None = None,
) -> tuple[bool, str, str]:
    if not dominio:
        return False, "", ""

    if lista_personal:
        for dom_user in lista_personal:
            d = dom_user.strip().lower().removeprefix("www.")
            if d and (dominio == d or dominio.endswith("." + d)):
                return True, "Tu Lista Personal", "Personal"

    for dom_verificado, categoria in lista_blanca.items():
        if dominio == dom_verificado or dominio.endswith("." + dom_verificado):
            return True, f"Lista Verificada ({categoria})", categoria

    extensiones_seguras = (".gob.es", ".gov", ".edu", ".edu.es", ".ac.uk")
    if dominio.endswith(extensiones_seguras):
        return True, "Dominio Institucional / Académico", "Institucional"

    return False, "", ""


def es_recomendado_para_perfil(categoria: str, dominio: str, perfil: str) -> bool:
    if not perfil or perfil == "general":
        return False
    cat = (categoria or "").lower()
    dom = (dominio or "").lower()
    if perfil == "estudiante":
        return any(k in cat for k in ["ciencia", "académ", "enciclopedia", "divulgación", "referencia"]) or dom.endswith((".edu", ".edu.es", ".ac.uk"))
    if perfil == "profesional":
        return any(k in cat for k in ["oficial", "institucional", "agencia", "prensa", "datos"]) or dom.endswith((".gob.es", ".gov"))
    if perfil == "investigacion":
        return any(k in cat for k in ["ciencia", "médica", "salud", "espacio", "científ"]) or dom.endswith((".edu", ".gov"))
    if perfil == "actualidad":
        return any(k in cat for k in ["verificación", "agencia", "prensa", "internacional"])
    return False


def buscar_en_web(
    consulta: str,
    num_resultados: int = 10,
    region: str = "wt-wt",
    idioma: str = "es",
) -> list[dict]:
    clave_cache = (consulta.lower().strip(), num_resultados, region, idioma)
    ahora = time.time()
    with _CACHE_LOCK:
        if clave_cache in _CACHE_BUSQUEDAS:
            ts, datos_guardados = _CACHE_BUSQUEDAS[clave_cache]
            if ahora - ts < _CACHE_TTL:
                return [dict(item) for item in datos_guardados]

    mapa_region = {
        "es": "es-es", "en": "us-en", "ca": "es-es", "gl": "es-es",
        "eu": "es-es", "fr": "fr-fr", "pt": "pt-pt", "de": "de-de", "it": "it-it",
    }
    region_respaldo = mapa_region.get(idioma, "us-en")
    region_efectiva = region_respaldo if region.endswith("-wt") else region

    intentos = [
        (region_efectiva, "yahoo"),
        (region_efectiva, "google"),
        (region_efectiva, "auto"),
        (region_efectiva, "bing"),
        (region_efectiva, "duckduckgo"),
    ]

    resultados = []
    for reg, bk in intentos:
        try:
            with DDGS(timeout=3) as ddgs:
                res = ddgs.text(consulta, region=reg, max_results=num_resultados, backend=bk)
                if res:
                    resultados = res
                    break
        except Exception:
            continue

    resultados_limpios = []
    for r in (resultados or []):
        url = r.get("href", "")
        if url:
            resultados_limpios.append({
                "titulo": r.get("title", "Sin título"),
                "url": url,
                "resumen": r.get("body", ""),
                "dominio": extraer_dominio(url),
            })

    if resultados_limpios:
        with _CACHE_LOCK:
            if len(_CACHE_BUSQUEDAS) >= 40:
                _CACHE_BUSQUEDAS.clear()
            _CACHE_BUSQUEDAS[clave_cache] = (ahora, [dict(item) for item in resultados_limpios])

    return resultados_limpios


def extraer_datos_web(url: str, dominio_propio: str) -> dict:
    """Descarga máx. 450 KB con timeout estricto y extrae texto + metadatos CRAAP."""
    ahora = time.time()
    with _CACHE_LOCK:
        if url in _CACHE_WEBS:
            ts, cache_item = _CACHE_WEBS[url]
            if ahora - ts < _CACHE_TTL:
                return dict(cache_item)

    vacio = {"texto": "", "autor": "", "fecha": "", "enlaces_externos": 0, "citas_fiables": False}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )
    }
    try:
        # Timeout estricto: 2s conectar, 2.5s leer; stream para no descargar webs enormes
        resp = requests.get(url, headers=headers, timeout=(2.0, 2.5), stream=True)
        resp.raise_for_status()

        tipo_contenido = resp.headers.get("Content-Type", "").lower()
        if "text/html" not in tipo_contenido:
            resp.close()
            return vacio

        chunks = []
        total_bytes = 0
        max_bytes = 450_000
        for chunk in resp.iter_content(chunk_size=32768):
            if chunk:
                chunks.append(chunk)
                total_bytes += len(chunk)
                if total_bytes >= max_bytes:
                    break
        resp.close()

        raw_bytes = b"".join(chunks)
        encoding = resp.encoding or "utf-8"
        html_raw = raw_bytes.decode(encoding, errors="replace")

        texto = trafilatura.extract(html_raw, include_comments=False, include_tables=False) or ""

        autor = ""
        fecha = ""
        try:
            meta = trafilatura.extract_metadata(html_raw)
            if meta:
                autor = (meta.author or "").strip()[:60]
                fecha = (meta.date or "").strip()[:20]
        except Exception:
            pass

        # Solo ejecutamos BeautifulSoup si trafilatura sacó poco texto
        if len(texto.split()) < 40:
            soup = BeautifulSoup(html_raw, "html.parser")
            parrafos = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            texto_bs = " ".join(parrafos)
            texto = texto_bs if len(texto_bs.split()) >= 40 else ""

        # Conteo ultrarrápido de enlaces externos con regex (sin parsear todo el DOM)
        enlaces_ext = set()
        citas_fiables = False
        for href in re.findall(r'href=["\'](https?://[^"\'>\s]+)', html_raw, flags=re.IGNORECASE):
            dom_link = extraer_dominio(href)
            if dom_link and dom_link != dominio_propio and not dom_link.endswith("." + dominio_propio):
                enlaces_ext.add(dom_link)
                if any(ref in dom_link for ref in DOMINIOS_REFERENCIA_CITAS):
                    citas_fiables = True

        resultado = {
            "texto": texto,
            "autor": autor,
            "fecha": fecha,
            "enlaces_externos": len(enlaces_ext),
            "citas_fiables": citas_fiables,
        }

        with _CACHE_LOCK:
            if len(_CACHE_WEBS) >= 150:
                _CACHE_WEBS.clear()
            _CACHE_WEBS[url] = (ahora, dict(resultado))

        return resultado
    except Exception:
        return vacio


def traducir_a_espanol(texto: str) -> tuple[str, bool]:
    fragmento = texto[:2200].strip()  # ~350 palabras: más rápido y suficiente para TF-IDF
    if not fragmento:
        return "", False
    try:
        traducido = GoogleTranslator(source="auto", target="es").translate(fragmento)
        return (traducido if traducido else fragmento), bool(traducido)
    except Exception:
        return fragmento, False


def preparar_texto_para_ia(item: dict, idioma: str) -> dict:
    datos_web = extraer_datos_web(item["url"], item["dominio"])
    texto_web = datos_web["texto"]

    if texto_web:
        texto_base = f"{item['titulo']}. {texto_web}"
        tipo = "Artículo completo"
    else:
        texto_base = f"{item['titulo']}. {item['resumen']}"
        tipo = "Solo resumen"

    texto_para_ia = texto_base
    nota = tipo
    if idioma != "es":
        traducido, ok = traducir_a_espanol(texto_base)
        if ok:
            texto_para_ia = traducido
            nota = f"{tipo} · Traducido a ES"
        else:
            nota = f"{tipo} · Sin traducir"

    return {
        "texto_original": texto_base,
        "texto_para_ia": texto_para_ia,
        "nota": nota,
        "tuvo_articulo": bool(texto_web),
        "autor": datos_web["autor"],
        "fecha": datos_web["fecha"],
        "enlaces_externos": datos_web["enlaces_externos"],
        "citas_fiables": datos_web["citas_fiables"],
    }


def calcular_craap_y_xai(
    titulo: str,
    prep: dict,
    texto_limpio: str,
    url: str,
    modelo,
) -> tuple[float, dict]:
    texto_orig = prep["texto_original"]
    num_palabras = len(texto_limpio.split())
    es_https = url.startswith("https://")
    exclamaciones = titulo.count("!") + texto_orig[:500].count("!")
    mayusculas_tit = sum(1 for w in titulo.split() if len(w) > 3 and w.isupper())
    tono_sobrio = exclamaciones <= 2 and mayusculas_tit <= 2

    texto_low = texto_orig.lower()
    tiene_clickbait_grave = any(pat in texto_low for pat in PATRONES_SENSACIONALISTAS)
    tiene_cifras = bool(re.search(r"\d+(?:[.,]\d+)?\s*%|\b(?:19|20)\d{2}\b", texto_orig))
    tiene_autor = bool(prep["autor"])
    tiene_fecha = bool(prep["fecha"])
    enlaces_ext = prep["enlaces_externos"]
    citas_fiables = prep["citas_fiables"]

    score = 0.35
    if es_https:
        score += 0.08
    if tiene_autor:
        score += 0.14
    if tiene_fecha:
        score += 0.12
    if num_palabras >= 150:
        score += 0.10
    if tiene_cifras:
        score += 0.10
    if citas_fiables:
        score += 0.15
    elif enlaces_ext >= 2:
        score += 0.08

    if not tono_sobrio:
        score -= 0.18
    if tiene_clickbait_grave:
        score -= 0.25

    score_craap = max(0.05, min(1.0, score))

    palabras_clave = []
    try:
        tfidf = modelo.named_steps["tfidf"]
        clf = modelo.named_steps["clf"]
        vec = tfidf.transform([texto_limpio])
        indices = vec.nonzero()[1]
        vocab = tfidf.get_feature_names_out()

        importancias = getattr(clf, "feature_importances_", None)
        if importancias is None and hasattr(clf, "named_estimators_") and "rf" in clf.named_estimators_:
            importancias = getattr(clf.named_estimators_["rf"], "feature_importances_", None)

        puntuados = []
        for idx in indices:
            peso = float(vec[0, idx])
            if importancias is not None:
                peso *= (float(importancias[idx]) + 1e-5)
            puntuados.append((peso, vocab[idx]))

        puntuados.sort(reverse=True)
        vistos = set()
        for _, term in puntuados:
            if term not in vistos and len(term) > 3:
                vistos.add(term)
                palabras_clave.append(term)
            if len(palabras_clave) == 5:
                break
    except Exception:
        palabras_clave = []

    xai = {
        "https": es_https,
        "extension_adecuada": num_palabras >= 150,
        "tono_sobrio": tono_sobrio and not tiene_clickbait_grave,
        "aporta_datos": tiene_cifras,
        "autor": prep["autor"],
        "fecha": prep["fecha"],
        "enlaces_externos": enlaces_ext,
        "citas_fiables": citas_fiables,
        "score_craap": round(score_craap, 3),
        "palabras_clave": palabras_clave,
    }
    return score_craap, xai


def generar_sintesis(consulta: str, aprobados: list[dict]) -> list[dict]:
    palabras_consulta = [
        w.lower() for w in re.findall(r"[a-záéíóúüñA-Z0-9]+", consulta) if len(w) >= 3
    ]
    candidatas = []

    for idx, item in enumerate(aprobados[:5], start=1):
        texto_fuente = item.get("_texto_bruto") or item.get("resumen", "")
        if not texto_fuente:
            continue

        frases = re.split(r"(?<=[.!?])\s+", texto_fuente)
        for frase in frases:
            frase_limpia = re.sub(r"\s+", " ", frase).strip()
            if len(frase_limpia) < 65 or len(frase_limpia) > 280:
                continue
            frase_low = frase_limpia.lower()
            coincidencias = sum(1 for p in palabras_consulta if p in frase_low)
            bonus_capa1 = 1.5 if item.get("fiabilidad") == 1.0 else 0.0
            bonus_datos = 0.5 if re.search(r"\d", frase_limpia) else 0.0
            score = coincidencias * 2.0 + bonus_capa1 + bonus_datos
            if score > 0:
                candidatas.append((score, idx, frase_limpia, item["dominio"], item["url"]))

    candidatas.sort(key=lambda x: x[0], reverse=True)

    sintesis = []
    fuentes_usadas = {}
    frases_vistas = set()

    for _, idx, frase, dom, url in candidatas:
        clave = frase[:45].lower()
        if clave in frases_vistas:
            continue
        if fuentes_usadas.get(idx, 0) >= 2:
            continue
        frases_vistas.add(clave)
        fuentes_usadas[idx] = fuentes_usadas.get(idx, 0) + 1
        sintesis.append({"texto": frase, "cita": idx, "dominio": dom, "url": url})
        if len(sintesis) == 3:
            break

    return sintesis


def buscar_y_filtrar(
    consulta: str,
    modelo,
    lista_blanca: dict,
    umbral: float = 0.55,
    num_resultados: int = 10,
    region: str = "wt-wt",
    idioma: str = "es",
    lista_personal: list[str] | None = None,
    perfil: str = "general",
) -> dict:
    t0 = time.time()
    resultados = buscar_en_web(consulta, num_resultados=num_resultados, region=region, idioma=idioma)
    t_ddgs = time.time() - t0

    aprobados = []
    filtrados = []
    pendientes_ia = []

    for item in resultados:
        es_verificada, motivo_capa1, categoria = comprobar_capa_1(
            item["dominio"], lista_blanca, lista_personal
        )
        if es_verificada:
            item["fiabilidad"] = 1.0
            item["metodo"] = motivo_capa1
            item["categoria"] = categoria
            item["es_personal"] = categoria == "Personal"
            item["recomendado_perfil"] = es_recomendado_para_perfil(categoria, item["dominio"], perfil)
            item["palabras_analizadas"] = None
            item["xai"] = {
                "https": item["url"].startswith("https://"),
                "extension_adecuada": True,
                "tono_sobrio": True,
                "aporta_datos": True,
                "autor": "",
                "fecha": "",
                "enlaces_externos": 0,
                "citas_fiables": True,
                "score_ml": 1.0,
                "score_craap": 1.0,
                "palabras_clave": [],
            }
            item["_texto_bruto"] = item.get("resumen", "")
            aprobados.append(item)
        else:
            pendientes_ia.append(item)

    t1 = time.time()
    capa1_para_texto = aprobados[:2]
    preparados = []

    if capa1_para_texto or pendientes_ia:
        executor = ThreadPoolExecutor(max_workers=12)
        futs_c1 = [
            executor.submit(extraer_datos_web, it["url"], it["dominio"])
            for it in capa1_para_texto
        ]
        futs_c2 = [
            executor.submit(preparar_texto_para_ia, it, idioma)
            for it in pendientes_ia
        ]
        # Tope duro de 3,5 segundos: si una web se queda colgada, pasamos al resumen sin esperarla
        wait(futs_c1 + futs_c2, timeout=3.5)
        executor.shutdown(wait=False, cancel_futures=True)

        for it, fut in zip(capa1_para_texto, futs_c1):
            if fut.done() and not fut.exception():
                dat = fut.result()
                if dat["texto"]:
                    it["_texto_bruto"] = dat["texto"]

        for item, fut in zip(pendientes_ia, futs_c2):
            if fut.done() and not fut.exception():
                preparados.append(fut.result())
            else:
                base = f"{item['titulo']}. {item['resumen']}"
                preparados.append({
                    "texto_original": base,
                    "texto_para_ia": base,
                    "nota": "Solo resumen (web lenta)",
                    "tuvo_articulo": False,
                    "autor": "",
                    "fecha": "",
                    "enlaces_externos": 0,
                    "citas_fiables": False,
                })

    t_webs = time.time() - t1
    t2 = time.time()

    if pendientes_ia:
        for item, prep in zip(pendientes_ia, preparados):
            texto_limpio = limpiar_texto(prep["texto_para_ia"])
            num_palabras = len(texto_limpio.split())
            item["palabras_analizadas"] = num_palabras
            item["es_personal"] = False
            item["recomendado_perfil"] = False
            item["categoria"] = "Web analizada por IA + CRAAP"
            item["_texto_bruto"] = prep["texto_original"]

            score_craap, xai = calcular_craap_y_xai(
                item["titulo"], prep, texto_limpio, item["url"], modelo
            )

            if num_palabras < 25:
                xai["score_ml"] = 0.0
                item["xai"] = xai
                item["fiabilidad"] = 0.0
                item["metodo"] = f"IA: texto insuficiente ({num_palabras} palabras)"
                filtrados.append(item)
                continue

            prob_ml = float(modelo.predict_proba([texto_limpio])[0][1])
            xai["score_ml"] = round(prob_ml, 3)
            item["xai"] = xai

            fiabilidad_final = (0.55 * prob_ml) + (0.45 * score_craap)
            item["fiabilidad"] = fiabilidad_final
            item["metodo"] = f"IA + CRAAP ({prep['nota']})"

            if fiabilidad_final >= umbral:
                aprobados.append(item)
            else:
                filtrados.append(item)

    aprobados.sort(
        key=lambda x: (
            1 if x.get("es_personal") else 0,
            x.get("fiabilidad") or 0.0,
            1 if x.get("recomendado_perfil") else 0,
        ),
        reverse=True,
    )

    sintesis = generar_sintesis(consulta, aprobados)
    for it in aprobados + filtrados:
        it.pop("_texto_bruto", None)

    t_ia = time.time() - t2
    print(f"⏱️ '{consulta}' -> DDGS: {t_ddgs:.2f}s | Webs: {t_webs:.2f}s | IA+CRAAP: {t_ia:.2f}s", flush=True)

    return {"aprobados": aprobados, "ocultos": filtrados, "sintesis": sintesis}