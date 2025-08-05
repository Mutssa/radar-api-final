print("--- O script crawler.py começou a ser executado ---")
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re, time, hashlib
import asyncio
import asyncpg

# --- CONFIGURAÇÕES GERAIS ---
MAX_PAGES_PER_DOMAIN = 80
CRAWL_DEPTH = 2
RATE_LIMIT_SECONDS = 1.2
_last_fetch = {}

# ==============================================================================
# CÓDIGO DE SCRAPING (AGORA COMPLETO)
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
    """Extrai cards de imóveis de uma página web"""
    cards = []
    
    # Seletores comuns para cards de imóveis
    selectors = [
        '.property-card', '.imovel-card', '.listing-card',
        '.property-item', '.imovel-item', '.listing-item',
        '[data-testid*="property"]', '[data-testid*="listing"]',
        '.card', '.item', '.result'
    ]
    
    for selector in selectors:
        elements = soup.select(selector)
        if elements:
            print(f"Encontrados {len(elements)} elementos com seletor: {selector}")
            break
    
    if not elements:
        # Fallback: procurar por qualquer elemento que contenha preço
        elements = soup.find_all(text=re.compile(r'R\$\s*[\d.,]+'))
        elements = [elem.parent for elem in elements if elem.parent]
        print(f"Fallback: Encontrados {len(elements)} elementos com preços")
    
    for element in elements:
        try:
            card = extract_property_data(element, base_url)
            if card and card.get('price'):  # Só adiciona se tiver pelo menos preço
                cards.append(card)
        except Exception as e:
            print(f"Erro ao extrair dados do card: {e}")
            continue
    
    return cards

def extract_property_data(element, base_url):
    """Extrai dados de um elemento de imóvel"""
    try:
        # Título
        title_selectors = ['h1', 'h2', 'h3', '.title', '.name', '[data-testid*="title"]']
        title = extract_text_by_selectors(element, title_selectors)
        
        # Preço
        price_text = extract_text_by_patterns(element, [r'R\$\s*[\d.,]+'])
        price = parse_price(price_text)
        
        # Tipo
        type_text = extract_text_by_patterns(element, [
            r'\b(apartamento|casa|sobrado|cobertura|loft|studio|kitnet)\b'
        ])
        
        # Área
        area_text = extract_text_by_patterns(element, [r'(\d+)\s*m²'])
        area = int(area_text.split()[0]) if area_text and area_text.split()[0].isdigit() else None
        
        # Quartos
        bedrooms_text = extract_text_by_patterns(element, [r'(\d+)\s*(quarto|dorm)'])
        bedrooms = int(bedrooms_text.split()[0]) if bedrooms_text and bedrooms_text.split()[0].isdigit() else None
        
        # Suítes
        suites_text = extract_text_by_patterns(element, [r'(\d+)\s*(suíte|suite)'])
        suites = int(suites_text.split()[0]) if suites_text and suites_text.split()[0].isdigit() else None
        
        # Vagas
        parking_text = extract_text_by_patterns(element, [r'(\d+)\s*(vaga|garagem)'])
        parking = int(parking_text.split()[0]) if parking_text and parking_text.split()[0].isdigit() else None
        
        # Localização
        location_selectors = ['.address', '.location', '.neighborhood', '[data-testid*="address"]']
        location = extract_text_by_selectors(element, location_selectors)
        
        # URL do imóvel
        link_element = element.find('a')
        url = None
        if link_element and link_element.get('href'):
            url = urljoin(base_url, link_element['href'])
        
        # Calcular completude
        fields = [title, price, type_text, area, bedrooms, location, url]
        completeness = sum(1 for field in fields if field) * 100 // len(fields)
        
        return {
            'id': generate_id(title, price, location),
            'title': title,
            'price': price,
            'type': type_text,
            'area': area,
            'bedrooms': bedrooms,
            'suites': suites,
            'parking': parking,
            'location': location,
            'lat': None,
            'lng': None,
            'url': url,
            'source': urlparse(base_url).netloc,
            'completeness': completeness,
            'method': 'crawler'
        }
    except Exception as e:
        print(f"Erro ao extrair dados do imóvel: {e}")
        return None

def extract_text_by_selectors(element, selectors):
    """Extrai texto usando uma lista de seletores CSS"""
    for selector in selectors:
        found = element.select_one(selector)
        if found:
            return found.get_text(strip=True)
    return None

def extract_text_by_patterns(element, patterns):
    """Extrai texto usando padrões regex"""
    text = element.get_text()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

def parse_price(price_text):
    """Converte texto de preço para número"""
    if not price_text:
        return None
    # Remove tudo exceto dígitos e vírgulas/pontos
    cleaned = re.sub(r'[^\d.,]', '', price_text)
    # Substitui vírgulas por pontos e converte
    try:
        return float(cleaned.replace(',', '.'))
    except:
        return None

def generate_id(title, price, location):
    """Gera um ID único para o imóvel"""
    content = f"{title}_{price}_{location}".lower()
    return hashlib.md5(content.encode()).hexdigest()[:16]

def dedupe_imoveis(imoveis):
    """Remove imóveis duplicados baseado em similaridade"""
    seen = {}
    for imovel in imoveis:
        # Cria uma chave baseada em título, preço e localização
        key_parts = [
            str(imovel.get('title', '')).lower().strip(),
            str(imovel.get('price', '')),
            str(imovel.get('location', '')).lower().strip()
        ]
        key = '|'.join(key_parts)
        
        # Se não viu ainda, ou se o atual tem mais completude, mantém
        if key not in seen or imovel.get('completeness', 0) > seen[key].get('completeness', 0):
            seen[key] = imovel
    
    return list(seen.values())

def varrer_e_extrair(start_url, filtros=None):
    print(f"\nIniciando varredura de domínio: {start_url}")
    try:
        pages = crawl_domain(start_url)
        print(f"Domínio varrido. {len(pages)} páginas encontradas.")
        
        todos = []
        for url, soup in pages:
            print(f"Extraindo dados de: {url}")
            cards = extract_cards_from_soup(soup, url)
            todos.extend(cards)
            print(f"  -> {len(cards)} cards encontrados nesta página")
        
        print(f"Extração concluída. {len(todos)} cards de imóveis encontrados.")
        deduped = dedupe_imoveis(todos)
        print(f"Imóveis únicos após deduplicação: {len(deduped)}")
        return deduped
    except Exception as e:
        print(f"Erro durante varredura de {start_url}: {e}")
        return []

# ==============================================================================
# CÓDIGO PARA SALVAR NO BANCO DE DADOS
# ==============================================================================

async def salvar_dados_no_banco(lista_de_imoveis, database_url):
    if not database_url:
        print("Erro: A URL do banco de dados não foi definida.")
        return

    try:
        conn = await asyncpg.connect(database_url)
        print("Conexão com banco estabelecida com sucesso.")
        
        # Criar tabela se não existir
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS imoveis (
                id TEXT PRIMARY KEY, 
                title TEXT, 
                price NUMERIC, 
                type TEXT,
                area INTEGER, 
                bedrooms INTEGER, 
                suites INTEGER, 
                parking INTEGER,
                location TEXT, 
                lat NUMERIC, 
                lng NUMERIC, 
                url TEXT,
                source TEXT, 
                completeness INTEGER, 
                method TEXT
            )
        ''')

        # Limpar dados anteriores
        await conn.execute("DELETE FROM imoveis")
        print("Dados anteriores removidos da tabela.")
        
        if not lista_de_imoveis:
            print("Nenhum imóvel para salvar.")
            await conn.close()
            return
        
        # Preparar dados para inserção
        colunas = [
            "id", "title", "price", "type", "area", "bedrooms", "suites",
            "parking", "location", "lat", "lng", "url", "source",
            "completeness", "method"
        ]
        
        registros_para_salvar = []
        for imovel in lista_de_imoveis:
            # Garantir que todos os campos existam
            imovel_completo = {}
            for key in colunas:
                imovel_completo[key] = imovel.get(key)
            
            # Garantir que há um ID
            if not imovel_completo.get('id'):
                imovel_completo['id'] = generate_id(
                    imovel_completo.get('title'),
                    imovel_completo.get('price'),
                    imovel_completo.get('location')
                )
            
            registros_para_salvar.append(tuple(imovel_completo.values()))
        
        # Inserir dados
        await conn.copy_records_to_table('imoveis', records=registros_para_salvar)
        await conn.close()
        print(f"Sucesso! {len(lista_de_imoveis)} imóveis foram salvos no banco de dados.")
        
    except Exception as e:
        print(f"Erro ao salvar no banco de dados: {e}")
        raise

# ==============================================================================
# SEÇÃO DE INICIALIZAÇÃO
# ==============================================================================

async def criar_tabela_links_se_necessario(database_url):
    """Cria a tabela 'links' se ela não existir"""
    try:
        conn = await asyncpg.connect(database_url)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS links (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                ativo BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await conn.close()
        print("Tabela 'links' verificada/criada com sucesso.")
    except Exception as e:
        print(f"Erro ao criar tabela 'links': {e}")
        raise

async def buscar_urls_para_varrer(database_url):
    """Conecta no banco e busca a lista de URLs da tabela 'links'."""
    print("Buscando lista de URLs no banco de dados...")
    try:
        # Primeiro, garantir que a tabela existe
        await criar_tabela_links_se_necessario(database_url)
        
        conn = await asyncpg.connect(database_url)
        
        # Verificar se há URLs na tabela
        count = await conn.fetchval('SELECT COUNT(*) FROM links WHERE ativo = TRUE')
        if count == 0:
            print("Aviso: Tabela 'links' está vazia. Adicionando URLs de exemplo...")
            # Inserir algumas URLs de exemplo
            urls_exemplo = [
                'https://www.vivareal.com.br',
                'https://www.zapimoveis.com.br',
                'https://www.imovelweb.com.br'
            ]
            for url in urls_exemplo:
                await conn.execute('INSERT INTO links (url) VALUES ($1)', url)
            print(f"Adicionadas {len(urls_exemplo)} URLs de exemplo.")
        
        # Buscar URLs ativas
        rows = await conn.fetch('SELECT url FROM links WHERE ativo = TRUE')
        await conn.close()
        
        urls = [row['url'] for row in rows]
        print(f"Encontradas {len(urls)} URLs ativas para varrer.")
        return urls
        
    except Exception as e:
        print(f"Erro ao buscar URLs do banco: {e}")
        raise

if __name__ == "__main__":
    # ------------------- CORRIJA ESTA LINHA (REMOVA O ESPAÇO) -------------------
    SUA_CONNECTION_STRING_DO_SUPABASE = "postgresql://postgres.ajayopsrusrasvzppmrq:RadarImob2025@aws-0-us-east-1.pooler.supabase.com:6543/postgres?ssl=require"
    # ----------------------------------------------------------------------------
    
    try:
        print("ETAPA 1: Tentando buscar URLs do Supabase...")
        urls_a_varrer = asyncio.run(buscar_urls_para_varrer(SUA_CONNECTION_STRING_DO_SUPABASE))
        
        print(f"ETAPA 2: Sucesso! Encontradas {len(urls_a_varrer)} URLs. Iniciando o scraper...")
        
        if not urls_a_varrer:
            print("Nenhuma URL encontrada no banco de dados para varrer.")
        else:
            todos_os_imoveis = []
            
            for i, url in enumerate(urls_a_varrer, 1):
                print(f"\n--- ETAPA 3.{i}: Varrendo URL: {url} ---")
                imoveis_do_site = varrer_e_extrair(url)
                todos_os_imoveis.extend(imoveis_do_site)
                print(f"Total acumulado: {len(todos_os_imoveis)} imóveis")

            print(f"\nETAPA 4: Varredura concluída. Total: {len(todos_os_imoveis)} imóveis encontrados.")
            
            if todos_os_imoveis:
                print("Salvando no banco de dados...")
                asyncio.run(salvar_dados_no_banco(todos_os_imoveis, SUA_CONNECTION_STRING_DO_SUPABASE))
            else:
                print("Nenhum imóvel encontrado nos sites.")

    except Exception as e:
        print(f"\nERRO GERAL: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- O script crawler.py terminou a execução. ---")