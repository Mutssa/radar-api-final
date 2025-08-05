print("--- O script crawler.py começou a ser executado ---")
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re, time
import asyncio
import asyncpg

# --- CONFIGURAÇÕES GERAIS ---
MAX_PAGES_PER_DOMAIN = 80
CRAWL_DEPTH = 2
RATE_LIMIT_SECONDS = 1.2
_last_fetch = {}

# ==============================================================================
# SEU CÓDIGO DE SCRAPING (NENHUMA MUDANÇA NECESSÁRIA AQUI)
# ==============================================================================

def fetch_rendered(url, timeout=10000):
    domain = urlparse(url).netloc.lower()
    last = _last_fetch.get(domain, 0)
    elapsed = time.time() - last
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    content = ""
    print(f"Buscando URL: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=timeout)
            time.sleep(0.8)
            content = page.content()
        except Exception as e:
            print(f"Erro ao buscar {url}: {e}")
            pass
        browser.close()
    _last_fetch[domain] = time.time()
    return content

def normalize_url(base, href):
    if not href:
        return None
    full = urljoin(base, href.split("#")[0])
    return full

def is_internal_link(base_domain, href):
    if not href:
        return False
    parsed = urlparse(href)
    if parsed.netloc and parsed.netloc.lower() != base_domain:
        return False
    return True

def crawl_domain(start_url):
    parsed_base = urlparse(start_url)
    domain = parsed_base.netloc.lower()
    visited = set()
    queue = [(start_url, 0)]
    pages = []

    while queue and len(pages) < MAX_PAGES_PER_DOMAIN:
        url, depth = queue.pop(0)
        if url in visited or depth > CRAWL_DEPTH:
            continue
        visited.add(url)
        html = fetch_rendered(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        pages.append((url, soup))
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not is_internal_link(domain, href):
                continue
            full = normalize_url(url, href)
            if not full or full in visited:
                continue
            priority_keywords = ["imovel", "venda", "aluguel", "buscar", "list", "pagina", "detalhe"]
            score = 0
            for kw in priority_keywords:
                if kw in full.lower() or (a.get_text() and kw in a.get_text().lower()):
                    score += 1
            if score > 0:
                queue.insert(0, (full, depth + 1))
            else:
                queue.append((full, depth + 1))
    return pages

def extract_cards_from_soup(soup, base_url):
    cards = []
    # ... seu código de extração continua aqui, está ótimo ...
    return cards

def dedupe_imoveis(imoveis):
    # ... seu código de deduplicação continua aqui, está ótimo ...
    return list(seen.values())

def varrer_e_extrair(start_url, filtros=None):
    print("\nIniciando varredura de domínio...")
    pages = crawl_domain(start_url)
    print(f"Domínio varrido. {len(pages)} páginas encontradas.")
    todos = []
    for url, soup in pages:
        cards = extract_cards_from_soup(soup, url)
        todos.extend(cards)
    print(f"Extração concluída. {len(todos)} cards de imóveis encontrados.")
    deduped = dedupe_imoveis(todos)
    print(f"Imóveis únicos após deduplicação: {len(deduped)}")
    return deduped

# ==============================================================================
# NOSSO CÓDIGO PARA SALVAR NO BANCO DE DADOS
# ==============================================================================

async def salvar_dados_no_banco(lista_de_imoveis, database_url):
    if not database_url:
        print("Erro: A URL do banco de dados não foi definida.")
        return

    conn = await asyncpg.connect(database_url)
    
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS imoveis (
            id TEXT PRIMARY KEY, title TEXT, price NUMERIC, type TEXT,
            area INTEGER, bedrooms INTEGER, suites INTEGER, parking INTEGER,
            location TEXT, lat NUMERIC, lng NUMERIC, url TEXT,
            source TEXT, completeness INTEGER, method TEXT
        )
    ''')

    await conn.execute("DELETE FROM imoveis")
    
    # Prepara os dados para inserção, garantindo que todas as colunas existam
    colunas = [
        "id", "title", "price", "type", "area", "bedrooms", "suites",
        "parking", "location", "lat", "lng", "url", "source",
        "completeness", "method"
    ]
    
    registros_para_salvar = []
    for imovel in lista_de_imoveis:
        # Garante que todos os campos existam no dicionário, mesmo que nulos
        imovel_completo = {key: imovel.get(key) for key in colunas}
        # Adiciona um ID único se não existir
        if not imovel_completo.get('id'):
            imovel_completo['id'] = imovel.get('link') or imovel.get('title')
        
        registros_para_salvar.append(imovel_completo)
        
    await conn.copy_records_to_table('imoveis', records=[tuple(r.values()) for r in registros_para_salvar])
    
    await conn.close()
    print(f"Sucesso! {len(lista_de_imoveis)} imóveis foram salvos no banco de dados.")

# ==============================================================================
# SEÇÃO DE INICIALIZAÇÃO (NOVA VERSÃO DINÂMICA)
# ==============================================================================

async def buscar_urls_para_varrer(database_url):
    """Conecta no banco e busca a lista de URLs da tabela 'links'."""
    print("Buscando lista de URLs no banco de dados...")
    conn = await asyncpg.connect(database_url)
    # Seleciona apenas a coluna 'url' da nossa tabela 'links'
    rows = await conn.fetch('SELECT url FROM links')
    await conn.close()
    # Extrai apenas o texto da URL de cada registro
    urls = [row['url'] for row in rows]
    return urls

if __name__ == "__main__":
    # ------------------- PREENCHA ESTA LINHA -------------------
    SUA_CONNECTION_STRING_DO_SUPABASE = "postgresql://postgres:[RadarImob2025]@db.ajayopsrusrasvzppmrq.supabase.co:5432/postgres" # <--- COLE SUA SENHA DO SUPABASE AQUI
    # --------------------------------------------------------------------
    
    try:
        # 1. Busca a lista de URLs da "agenda" no Supabase
        urls_a_varrer = asyncio.run(buscar_urls_para_varrer(SUA_CONNECTION_STRING_DO_SUPABASE))
        
        if not urls_a_varrer:
            print("Nenhuma URL encontrada no banco de dados para varrer. Adicione URLs primeiro.")
        else:
            print(f"Encontradas {len(urls_a_varrer)} URLs para varrer.")
            
            todos_os_imoveis = []
            # 2. Faz um loop, executando o scraper para cada URL encontrada
            for url in urls_a_varrer:
                print(f"\n{'='*20}\nIniciando varredura para: {url}\n{'='*20}")
                imoveis_do_site = varrer_e_extrair(url)
                todos_os_imoveis.extend(imoveis_do_site)

            # 3. Se encontrou imóveis no total, salva todos no banco de dados
            if todos_os_imoveis:
                print("\nEnviando todos os dados encontrados para o banco de dados na nuvem...")
                asyncio.run(salvar_dados_no_banco(todos_os_imoveis, SUA_CONNECTION_STRING_DO_SUPABASE))
            else:
                print("\nNenhum imóvel encontrado em nenhuma das fontes.")

    except Exception as e:
        print(f"\nOcorreu um erro geral durante a execução: {e}")

    print("\n--- O script crawler.py terminou a execução. ---")