# app.py - Radar Imobiliário Backend Completo
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl, EmailStr, validator
from typing import List, Optional, Dict, Any, Set
import asyncio
import aiohttp
import re
import json
import hashlib
import jwt
from datetime import datetime, timedelta
import os
import logging
from urllib.parse import urljoin, urlparse
import time
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
import redis.asyncio as redis
from contextlib import asynccontextmanager
import uuid
import asyncpg
from dataclasses import dataclass
import pickle

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "radar-secret-key-change-in-production")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    MAX_LINKS_PER_USER: int = int(os.getenv("MAX_LINKS_PER_USER", "30"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
    ENABLE_BOOTSTRAP_GPT4: bool = os.getenv("ENABLE_BOOTSTRAP_GPT4", "true").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

config = Config()

# Initialize OpenAI
if config.OPENAI_API_KEY:
    openai.api_key = config.OPENAI_API_KEY

# Global variables
db_pool = None
redis_client = None
rate_limiter = {}
users_db = {}
links_db = {}
properties_cache = {}
history_db = {}
semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting Radar Imobiliário API...")
    
    global redis_client
    try:
        redis_client = redis.from_url(config.REDIS_URL)
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}. Running without Redis cache.")
        redis_client = None
    
    global db_pool
    if config.DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(config.DATABASE_URL)
            logger.info("✅ Database connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Database connection failed: {e}. Running in memory.")
    
    logger.info("🎯 Radar Imobiliário API started successfully!")
    yield
    
    logger.info("🛑 Shutting down Radar Imobiliário API...")
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

app = FastAPI(
    title="🎯 Radar Imobiliário API",
    description="Sistema inteligente de extração e análise de imóveis usando IA",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if config.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if config.ENVIRONMENT == "development" else None
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Pydantic Models
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    
    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Nome deve ter pelo menos 2 caracteres')
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Senha deve ter pelo menos 6 caracteres')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class LinkCreate(BaseModel):
    url: HttpUrl
    name: Optional[str] = None
    
    @validator('url', pre=True)
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            v = 'https://' + v
        try:
            return HttpUrl(v)
        except ValueError:
            raise ValueError('URL inválida')

class PropertyFilter(BaseModel):
    type: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    bedrooms: Optional[int] = None
    suites: Optional[int] = None
    parking: Optional[int] = None
    min_area: Optional[float] = None
    location: Optional[str] = None
    
    @validator('min_price', 'max_price', 'min_area')
    def validate_positive_numbers(cls, v):
        if v is not None and v < 0:
            raise ValueError('Valores devem ser positivos')
        return v
    
    @validator('bedrooms', 'suites', 'parking')
    def validate_positive_integers(cls, v):
        if v is not None and v < 0:
            raise ValueError('Valores devem ser positivos')
        return v

class SearchRequest(BaseModel):
    filters: Optional[PropertyFilter] = None
    force_refresh: Optional[bool] = False

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        if len(v.strip()) < 1:
            raise ValueError('Mensagem não pode estar vazia')
        return v.strip()

class Property(BaseModel):
    id: str
    title: str
    price: float
    type: str
    area: Optional[float] = None
    bedrooms: Optional[int] = None
    suites: Optional[int] = None
    parking: Optional[int] = None
    location: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    url: str
    source: str
    images: List[str] = []
    features: List[str] = []
    completeness: int
    method: str
    extracted_at: datetime
    link_id: str

# Utility Functions
def hash_password(password: str) -> str:
    salt = hashlib.sha256(config.SECRET_KEY.encode()).hexdigest()[:16]
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def create_jwt_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    
    payload = {
        "user_id": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    return jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")

def verify_jwt_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    user_id = verify_jwt_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user = None
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        except Exception as e:
            logger.error(f"Database error: {e}")
    
    if not user and user_id in users_db:
        user = users_db[user_id]
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário não encontrado"
        )
    
    return dict(user) if hasattr(user, '_asdict') else user

async def check_rate_limit(user_id: str, action: str = "api_call") -> bool:
    if not redis_client:
        return True
    
    key = f"rate_limit:{user_id}:{action}"
    current_time = int(time.time())
    window_start = current_time - config.RATE_LIMIT_WINDOW
    
    try:
        await redis_client.zremrangebyscore(key, 0, window_start)
        current_count = await redis_client.zcard(key)
        
        if current_count >= config.RATE_LIMIT_REQUESTS:
            return False
        
        await redis_client.zadd(key, {str(uuid.uuid4()): current_time})
        await redis_client.expire(key, config.RATE_LIMIT_WINDOW)
        
        return True
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        return True

async def get_cached_data(key: str) -> Optional[Any]:
    if not redis_client:
        return None
    
    try:
        data = await redis_client.get(key)
        if data:
            return pickle.loads(data)
    except Exception as e:
        logger.error(f"Cache get error: {e}")
    
    return None

async def set_cached_data(key: str, data: Any, ttl: int = 3600) -> None:
    if not redis_client:
        return
    
    try:
        serialized = pickle.dumps(data)
        await redis_client.setex(key, ttl, serialized)
    except Exception as e:
        logger.error(f"Cache set error: {e}")

def apply_filters(properties: List[Dict], filters: PropertyFilter) -> List[Dict]:
    if not filters:
        return properties
    
    filtered = []
    for prop in properties:
        if filters.type and prop.get('type') != filters.type: continue
        price = prop.get('price', 0)
        if filters.min_price and price < filters.min_price: continue
        if filters.max_price and price > filters.max_price: continue
        if filters.bedrooms and prop.get('bedrooms', 0) < filters.bedrooms: continue
        if filters.suites and prop.get('suites', 0) < filters.suites: continue
        if filters.parking and prop.get('parking', 0) < filters.parking: continue
        if filters.min_area and prop.get('area', 0) < filters.min_area: continue
        if filters.location:
            location = prop.get('location', '').lower()
            if filters.location.lower() not in location: continue
        filtered.append(prop)
    return filtered

def deduplicate_properties(properties: List[Dict]) -> List[Dict]:
    seen = set()
    unique_properties = []
    
    for prop in properties:
        key_fields = f"{prop.get('title', '').lower()}-{prop.get('price', 0)}-{prop.get('location', '').lower()}"
        prop_hash = hashlib.md5(key_fields.encode()).hexdigest()
        
        if prop_hash not in seen:
            seen.add(prop_hash)
            unique_properties.append(prop)
    
    return unique_properties

def calculate_property_score(prop: Dict) -> float:
    score = 0
    score += prop.get('completeness', 0) * 0.3
    if prop.get('method') == 'bootstrap': score += 30
    else: score += 15
    extracted_at = prop.get('extracted_at')
    if extracted_at:
        hours_old = (datetime.utcnow() - extracted_at).total_seconds() / 3600
        score += max(0, 20 - hours_old)
    return score

# Web Scraping Classes
class PropertyExtractor:
    def __init__(self):
        self.session = None
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={'User-Agent': self.user_agents[0]}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def geocode_location(self, location: str) -> Optional[Dict]:
        if not location or location == "Localização não informada":
            return None
        cache_key = f"geocode:{hashlib.md5(location.encode()).hexdigest()}"
        cached_coords = await get_cached_data(cache_key)
        if cached_coords:
            return cached_coords
        try:
            url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
            headers = {'User-Agent': 'RadarImobiliarioApp/1.0 (contact@radarimobiliario.com)'}
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Geocoding failed for '{location}' with status {response.status}")
                    return None
                data = await response.json()
                if data and len(data) > 0:
                    coords = {'lat': float(data[0]['lat']), 'lng': float(data[0]['lon'])}
                    await set_cached_data(cache_key, coords, ttl=86400)
                    return coords
        except Exception as e:
            logger.error(f"Error during geocoding for '{location}': {e}")
        return None

    async def extract_properties(self, url: str, use_bootstrap: bool = True) -> List[Dict]:
        async with semaphore:
            try:
                if use_bootstrap and config.ENABLE_BOOTSTRAP_GPT4 and config.OPENAI_API_KEY:
                    properties = await self.extract_with_bootstrap(url)
                    if properties: return properties
                return await self.extract_with_heuristics(url)
            except Exception as e:
                logger.error(f"Extraction error for {url}: {e}")
                return []

    async def extract_with_heuristics(self, url: str) -> List[Dict]:
        try:
            async with self.session.get(url) as response:
                if response.status != 200: return []
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                properties_to_geocode = []
                domain = urlparse(url).netloc.lower()
                selectors = self.get_selectors_for_domain(domain)
                for selector_set in selectors:
                    containers = soup.select(selector_set.get('container', 'body'))
                    if not containers: continue
                    for i, container in enumerate(containers[:50]):
                        property_data = self.extract_property_from_container_sync(container, selector_set, url, i)
                        if property_data: properties_to_geocode.append(property_data)
                    if properties_to_geocode: break
                tasks = [self.geocode_property(prop) for prop in properties_to_geocode]
                geocoded_properties = await asyncio.gather(*tasks)
                return [prop for prop in geocoded_properties if prop]
        except Exception as e:
            logger.error(f"Heuristic extraction error for {url}: {e}")
            return []

    async def extract_with_bootstrap(self, url: str) -> List[Dict]:
        try:
            async with self.session.get(url) as response:
                if response.status != 200: return []
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
                sample_html = str(soup)[:8000]
                prompt = self.create_bootstrap_prompt(sample_html, url)
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                rules_text = response.choices[0].message.content.strip()
                try:
                    rules = json.loads(rules_text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', rules_text, re.DOTALL)
                    if json_match: rules = json.loads(json_match.group())
                    else:
                        logger.error(f"Could not parse GPT-4 response: {rules_text}")
                        return []
                properties_to_geocode = self.apply_bootstrap_rules_sync(soup, rules, url)
                tasks = [self.geocode_property(prop) for prop in properties_to_geocode]
                geocoded_properties = await asyncio.gather(*tasks)
                return [prop for prop in geocoded_properties if prop]
        except Exception as e:
            logger.error(f"Bootstrap extraction error for {url}: {e}")
        return []

    async def geocode_property(self, prop: Dict) -> Dict:
        coords = await self.geocode_location(prop['location'])
        if coords:
            prop['lat'] = coords['lat']
            prop['lng'] = coords['lng']
        return prop

    def extract_property_from_container_sync(self, container, selectors: Dict, base_url: str, index: int) -> Optional[Dict]:
        try:
            title_elem = container.select_one(selectors.get('title', ''))
            price_elem = container.select_one(selectors.get('price', ''))
            if not title_elem or not price_elem: return None
            title = title_elem.get_text(strip=True)
            price_text = price_elem.get_text(strip=True)
            price = self.extract_price(price_text)
            if not price or price <= 0: return None
            area = self.extract_area(container.get_text())
            bedrooms = self.extract_bedrooms(container.get_text())
            suites = max(0, (bedrooms or 1) - 1) if bedrooms else 0
            parking = self.extract_parking(container.get_text())
            location = self.extract_location(container.get_text()) or "Localização não informada"
            full_text = title + " " + price_text + " " + container.get_text()
            property_type = self.detect_property_type(full_text)
            link_elem = container.select_one(selectors.get('link', 'a[href]'))
            property_url = base_url
            if link_elem and link_elem.get('href'): property_url = urljoin(base_url, link_elem['href'])
            property_id = hashlib.md5(f"{base_url}-{title}-{price}-{index}".encode()).hexdigest()
            completeness = self.calculate_completeness(title, price, area, bedrooms, location)
            return {
                'id': property_id, 'title': title, 'price': price, 'type': property_type,
                'area': area, 'bedrooms': bedrooms, 'suites': suites, 'parking': parking or 1,
                'location': location, 'lat': None, 'lng': None, 'url': property_url,
                'source': urlparse(base_url).netloc, 'images': [], 'features': self.extract_features(container.get_text()),
                'completeness': completeness, 'method': 'heuristica', 'extracted_at': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error extracting property from container: {e}")
            return None
    
    def apply_bootstrap_rules_sync(self, soup: BeautifulSoup, rules: Dict, base_url: str) -> List[Dict]:
        properties = []
        try:
            containers = soup.select(rules.get('container_selector', 'body'))[:50]
            for i, container in enumerate(containers):
                property_data = self.extract_with_rules_sync(container, rules, base_url, i)
                if property_data:
                    property_data['method'] = 'bootstrap'
                    properties.append(property_data)
        except Exception as e:
            logger.error(f"Error applying bootstrap rules: {e}")
        return properties
        
    def extract_with_rules_sync(self, container, rules: Dict, base_url: str, index: int) -> Optional[Dict]:
        try:
            title_elem = container.select_one(rules.get('title_selector', ''))
            price_elem = container.select_one(rules.get('price_selector', ''))
            if not title_elem or not price_elem: return None
            title = title_elem.get_text(strip=True)
            price_text = price_elem.get_text(strip=True)
            price = self.extract_price(price_text)
            if not price or price <= 0: return None
            area_elem = container.select_one(rules.get('area_selector', ''))
            area = self.extract_area(area_elem.get_text() if area_elem else '')
            bedrooms_elem = container.select_one(rules.get('bedrooms_selector', ''))
            bedrooms = self.extract_bedrooms(bedrooms_elem.get_text() if bedrooms_elem else '')
            location_elem = container.select_one(rules.get('location_selector', ''))
            location = location_elem.get_text(strip=True) if location_elem else "Localização não informada"
            full_text = title + " " + price_text
            property_type = self.detect_type_with_rules(full_text, rules.get('type_indicators', {}))
            link_elem = container.select_one(rules.get('link_selector', 'a'))
            property_url = urljoin(base_url, link_elem.get('href', '')) if link_elem and link_elem.get('href') else base_url
            property_id = hashlib.md5(f"{base_url}-{title}-{price}-{index}".encode()).hexdigest()
            completeness = self.calculate_completeness(title, price, area, bedrooms, location)
            return {
                'id': property_id, 'title': title, 'price': price, 'type': property_type,
                'area': area, 'bedrooms': bedrooms, 'suites': max(0, (bedrooms or 1) - 1) if bedrooms else 0,
                'parking': 1, 'location': location, 'lat': None, 'lng': None, 'url': property_url,
                'source': urlparse(base_url).netloc, 'images': [], 'features': [],
                'completeness': completeness, 'method': 'bootstrap', 'extracted_at': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error extracting with rules: {e}")
            return None

    def create_bootstrap_prompt(self, html: str, url: str) -> str:
        return f"""
Você é um especialista em web scraping de sites imobiliários. Analise este HTML e retorne um JSON com seletores CSS para extrair informações de imóveis.

URL: {url}

Retorne APENAS um JSON válido com esta estrutura:
{{
    "container_selector": "seletor para cada card/item de imóvel",
    "title_selector": "seletor para título do imóvel",
    "price_selector": "seletor para preço",
    "area_selector": "seletor para área em m²",
    "bedrooms_selector": "seletor para número de quartos",
    "location_selector": "seletor para localização",
    "link_selector": "seletor para link do imóvel",
    "type_indicators": {{
        "sale": ["palavras que indicam venda"],
        "rent": ["palavras que indicam aluguel"]
    }}
}}

HTML:
{html}

Responda APENAS com o JSON, sem explicações:
"""

    def get_selectors_for_domain(self, domain: str) -> List[Dict]:
        selectors = {
            'vivareal.com.br': [{'container': '.property-card__container, .result-card', 'title': '.property-card__title, .result-card__title', 'price': '.property-card__price, .result-card__price', 'area': '[data-type="area"], .property-card__area', 'bedrooms': '[data-type="bedrooms"], .property-card__details-item:contains("quarto")', 'location': '.property-card__address, .result-card__address', 'link': 'a[href]'}],
            'zapimoveis.com.br': [{'container': '.result-card, .listing-card', 'title': '.result-card__title, .listing-card__title', 'price': '.result-card__price, .listing-card__price', 'area': '.result-card__area, .listing-card__area', 'bedrooms': '.result-card__bedrooms, .listing-card__bedrooms', 'location': '.result-card__address, .listing-card__address', 'link': 'a[href]'}],
            'quintoandar.com.br': [{'container': '[data-testid="listing-card"]', 'title': 'h2, [data-testid="listing-title"]', 'price': '[data-testid="price"], .price', 'area': '[data-testid="area"]', 'bedrooms': '[data-testid="bedrooms"]', 'location': '[data-testid="address"]', 'link': 'a[href]'}]
        }
        domain_selectors = []
        for site_domain, site_selectors in selectors.items():
            if site_domain in domain: domain_selectors.extend(site_selectors)
        generic_selectors = [{'container': '.card, .item, .listing, .property, [class*="card"], [class*="item"], [class*="property"]', 'title': 'h1, h2, h3, .title, [class*="title"]', 'price': '.price, [class*="price"], [class*="valor"]', 'area': '[class*="area"], [class*="m2"]', 'bedrooms': '[class*="quarto"], [class*="bedroom"], [class*="dorm"]', 'location': '.address, [class*="address"], .location, [class*="local"]', 'link': 'a[href]'}]
        return domain_selectors + generic_selectors

    def extract_price(self, text: str) -> Optional[float]:
        if not text: return None
        clean_text = re.sub(r'[R$\s]', '', text)
        numbers = re.findall(r'[\d.,]+', clean_text)
        if not numbers: return None
        largest_num = max(numbers, key=len)
        try:
            if ',' in largest_num and '.' in largest_num: largest_num = largest_num.replace('.', '').replace(',', '.')
            elif largest_num.count(',') == 1 and len(largest_num.split(',')[1]) == 2: largest_num = largest_num.replace(',', '.')
            elif ',' in largest_num: largest_num = largest_num.replace(',', '')
            price = float(largest_num)
            if 10000 <= price <= 100000000: return price
        except ValueError: pass
        return None

    def extract_area(self, text: str) -> Optional[float]:
        if not text: return None
        patterns = [r'(\d+(?:[.,]\d+)?)\s*m[²2]', r'(\d+(?:[.,]\d+)?)\s*metros', r'área[:\s]*(\d+(?:[.,]\d+)?)', r'(\d+(?:[.,]\d+)?)\s*m²']
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    area = float(matches[0].replace(',', '.'))
                    if 10 <= area <= 10000: return area
                except ValueError: continue
        return None

    def extract_bedrooms(self, text: str) -> Optional[int]:
        if not text: return None
        patterns = [r'(\d+)\s*(?:quartos?|dormitórios?|bedrooms?)', r'(\d+)\s*(?:qto|dorm|bed)', r'(\d+)\s*q']
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    bedrooms = int(matches[0])
                    if 0 <= bedrooms <= 20: return bedrooms
                except ValueError: continue
        return None

    def extract_parking(self, text: str) -> Optional[int]:
        if not text: return None
        patterns = [r'(\d+)\s*(?:vagas?|garagem|parking)', r'(\d+)\s*vaga']
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    parking = int(matches[0])
                    if 0 <= parking <= 10: return parking
                except ValueError: continue
        return 1

    def extract_location(self, text: str) -> Optional[str]:
        if not text: return None
        patterns = [r'([A-ZÁÊÇÕ][a-záêçõ\s]+-\s*[A-Z]{2})', r'([A-ZÁÊÇÕ][a-záêçõ\s]+,\s*[A-Z]{2})', r'([A-ZÁÊÇÕ][a-záêçõ\s]+\s*-\s*São Paulo)', r'([A-ZÁÊÇÕ][a-záêçõ\s]{3,})']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                location = matches[0].strip()
                if len(location) > 3: return location
        return None

    def extract_features(self, text: str) -> List[str]:
        features = []
        feature_keywords = ['piscina', 'academia', 'churrasqueira', 'varanda', 'sacada', 'elevador', 'portaria', 'segurança', 'playground', 'salão', 'jardim', 'quintal', 'terraço', 'cobertura', 'vista mar']
        text_lower = text.lower()
        for keyword in feature_keywords:
            if keyword in text_lower: features.append(keyword.title())
        return features[:5]

    def detect_property_type(self, text: str) -> str:
        text_lower = text.lower()
        rent_keywords = ['aluguel', 'alugar', 'locação', 'locar', 'rent', '/mês']
        sale_keywords = ['venda', 'vender', 'comprar', 'à venda', 'sale']
        rent_score = sum(1 for keyword in rent_keywords if keyword in text_lower)
        sale_score = sum(1 for keyword in sale_keywords if keyword in text_lower)
        return 'aluguel' if rent_score > sale_score else 'venda'

    def detect_type_with_rules(self, text: str, type_indicators: Dict) -> str:
        text_lower = text.lower()
        rent_indicators = type_indicators.get('rent', [])
        sale_indicators = type_indicators.get('sale', [])
        rent_score = sum(1 for indicator in rent_indicators if indicator.lower() in text_lower)
        sale_score = sum(1 for indicator in sale_indicators if indicator.lower() in text_lower)
        if rent_score > sale_score: return 'aluguel'
        elif sale_score > rent_score: return 'venda'
        else: return self.detect_property_type(text)

    def calculate_completeness(self, title: str, price: float, area: Optional[float], bedrooms: Optional[int], location: str) -> int:
        score = 0
        if title and len(title.strip()) > 5: score += 25
        if price and price > 0: score += 30
        if area and area > 0: score += 20
        if bedrooms and bedrooms > 0: score += 15
        if location and location != "Localização não informada" and len(location) > 5: score += 10
        return min(score, 100)

async def update_link_stats_and_history(link: Dict, old_properties_count: int, new_properties_count: int):
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """UPDATE links SET last_crawled = $1, properties_count = $2
                       WHERE id = $3""",
                    link['last_crawled'], link['properties_count'], link['id']
                )
                properties_added = new_properties_count - old_properties_count if new_properties_count > old_properties_count else 0
                properties_removed = old_properties_count - new_properties_count if old_properties_count > new_properties_count else 0
                await conn.execute(
                    """INSERT INTO extraction_history (link_id, properties_added, properties_removed)
                       VALUES ($1, $2, $3)""",
                    link['id'], properties_added, properties_removed
                )
    except Exception as e:
        logger.error(f"Error updating link stats and history: {e}")

# API Endpoints

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "2.0.0",
        "environment": config.ENVIRONMENT,
        "services": {}
    }
    
    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
            health_status["services"]["redis"] = "connected"
        else:
            health_status["services"]["redis"] = "not_configured"
    except Exception as e:
        health_status["services"]["redis"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Database
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["services"]["database"] = "connected"
        else:
            health_status["services"]["database"] = "not_configured"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check OpenAI
    health_status["services"]["openai"] = "configured" if config.OPENAI_API_KEY else "not_configured"
    
    return health_status
@app.post("/auth/register")
async def register_user(user_data: UserRegister):
    if db_pool:
        async with db_pool.acquire() as conn:
            existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", user_data.email)
            if existing: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email já cadastrado")
    elif user_data.email in [u['email'] for u in users_db.values()]: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email já cadastrado")
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(user_data.password)
    user_record = {"id": user_id, "name": user_data.name, "email": user_data.email, "password": hashed_password, "created_at": datetime.utcnow(), "is_active": True}
    if db_pool:
        try:
            async with db_pool.acquire() as conn: await conn.execute("""INSERT INTO users (id, name, email, password, created_at, is_active) VALUES ($1, $2, $3, $4, $5, $6)""", user_id, user_data.name, user_data.email, hashed_password, datetime.utcnow(), True)
        except Exception as e: logger.error(f"Database error during registration: {e}"); users_db[user_id] = user_record
    else: users_db[user_id] = user_record
    links_db[user_id] = []
    token = create_jwt_token(user_id)
    logger.info(f"New user registered: {user_data.email}")
    return {"message": "Usuário registrado com sucesso", "token": token, "user": {"id": user_id, "name": user_data.name, "email": user_data.email, "created_at": user_record["created_at"]}}

@app.post("/auth/login")
async def login_user(login_data: UserLogin):
    user = None
    if db_pool:
        try:
            async with db_pool.acquire() as conn: user = await conn.fetchrow("SELECT * FROM users WHERE email = $1 AND is_active = true", login_data.email)
        except Exception as e: logger.error(f"Database error during login: {e}")
    if not user:
        for u in users_db.values():
            if u['email'] == login_data.email and u.get('is_active', True): user = u; break
    if not user or not verify_password(login_data.password, user['password']): raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email ou senha incorretos")
    token = create_jwt_token(user['id'])
    logger.info(f"User logged in: {login_data.email}")
    return {"message": "Login realizado com sucesso", "token": token, "user": {"id": user['id'], "name": user['name'], "email": user['email']}}

@app.get("/links")
async def get_user_links(current_user: dict = Depends(get_current_user)):
    user_id = current_user['id']
    if db_pool:
        try:
            async with db_pool.acquire() as conn: links = await conn.fetch("SELECT id, url, name, properties_count, last_crawled FROM links WHERE user_id = $1 ORDER BY created_at DESC", user_id)
            return [dict(link) for link in links]
        except Exception as e: logger.error(f"Database error getting links: {e}"); return links_db.get(user_id, [])
    return links_db.get(user_id, [])

@app.post("/links")
async def add_link(link_data: LinkCreate, current_user: dict = Depends(get_current_user)):
    """Add a new link to user's collection"""
    user_id = current_user['id']
    
    if not await check_rate_limit(user_id, "add_link"):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Muitas requisições. Tente novamente em alguns minutos.")
    
    current_links = []
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                links_from_db = await conn.fetch("SELECT url FROM links WHERE user_id = $1", user_id)
                current_links = [link['url'] for link in links_from_db]
        except Exception as e:
            logger.error(f"Database error: {e}")
            current_links = [link['url'] for link in links_db.get(user_id, [])]
    else:
        current_links = [link['url'] for link in links_db.get(user_id, [])]
    
    if len(current_links) >= config.MAX_LINKS_PER_USER:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Limite máximo de {config.MAX_LINKS_PER_USER} links atingido")
    
    url_str = str(link_data.url)
    if url_str in current_links:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Link já adicionado")
    
    link_id = str(uuid.uuid4())
    domain = urlparse(url_str).netloc
    
    link_record = {
        "id": link_id, "user_id": user_id, "url": url_str, "name": link_data.name or domain,
        "domain": domain, "status": "active", "created_at": datetime.utcnow(),
        "last_crawled": None, "properties_count": 0
    }
    
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO links (id, user_id, url, name, domain, status, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    link_id, user_id, url_str, link_record["name"], domain, "active", datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Database error adding link: {e}")
            links_db.setdefault(user_id, []).append(link_record)
    else:
        links_db.setdefault(user_id, []).append(link_record)
    
    logger.info(f"Link added by user {user_id}: {url_str}")
    
    return {"message": "Link adicionado com sucesso", "link": link_record}

@app.delete("/links/{link_id}")
async def remove_link(link_id: str, current_user: dict = Depends(get_current_user)):
    """Remove a link from user's collection"""
    user_id = current_user['id']
    
    removed = False
    
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM links WHERE id = $1 AND user_id = $2",
                    link_id, user_id
                )
                removed = result == "DELETE 1"
        except Exception as e:
            logger.error(f"Database error removing link: {e}")
            
    if not removed:
        user_links = links_db.get(user_id, [])
        original_count = len(user_links)
        links_db[user_id] = [link for link in user_links if link['id'] != link_id]
        removed = len(links_db[user_id]) < original_count
    
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Link não encontrado"
        )
    
    cache_key = f"properties:{link_id}"
    if redis_client:
        try:
            await redis_client.delete(cache_key)
        except Exception:
            pass
    
    if link_id in properties_cache:
        del properties_cache[link_id]
    
    logger.info(f"Link removed by user {user_id}: {link_id}")
    
    return {"message": "Link removido com sucesso"}

@app.post("/search")
async def search_properties(
    search_request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user['id']
    if not await check_rate_limit(user_id, "search"):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Muitas buscas. Tente novamente em alguns minutos.")
    user_links = []
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                links = await conn.fetch("SELECT * FROM links WHERE user_id = $1 AND status = 'active'", user_id)
                user_links = [dict(link) for link in links]
        except Exception as e:
            logger.error(f"Database error: {e}")
            user_links = links_db.get(user_id, [])
    else:
        user_links = [link for link in links_db.get(user_id, []) if link.get('status') == 'active']
    if not user_links:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nenhum link ativo encontrado. Adicione links primeiro.")
    all_properties = []
    extraction_stats = {"total_links": len(user_links), "successful_extractions": 0, "failed_extractions": 0, "total_properties": 0}
    if not search_request.force_refresh:
        cached_properties = await get_cached_data(f"user_properties:{user_id}")
        if cached_properties:
            filtered_properties = apply_filters(cached_properties, search_request.filters)
            return {"total": len(filtered_properties), "properties": sorted(filtered_properties, key=calculate_property_score, reverse=True), "sources": len(user_links), "extracted_at": cached_properties[0]['extracted_at'] if cached_properties else datetime.utcnow(), "cached": True, "stats": extraction_stats}
    async with PropertyExtractor() as extractor:
        tasks = [extract_and_geocode_link(extractor, link) for link in user_links]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Extraction failed for {user_links[i]['url']}: {result}")
                extraction_stats["failed_extractions"] += 1
            else:
                properties, updated_link = result
                all_properties.extend(properties)
                extraction_stats["successful_extractions"] += 1
                extraction_stats["total_properties"] += len(properties)
                old_properties_count = user_links[i].get('properties_count', 0)
                background_tasks.add_task(update_link_stats_and_history, updated_link, old_properties_count, len(properties))
    unique_properties = deduplicate_properties(all_properties)
    await set_cached_data(f"user_properties:{user_id}", unique_properties, ttl=1800)
    filtered_properties = apply_filters(unique_properties, search_request.filters)
    sorted_properties = sorted(filtered_properties, key=calculate_property_score, reverse=True)
    logger.info(f"Search completed for user {user_id}: {len(sorted_properties)} properties found")
    return {"total": len(sorted_properties), "properties": sorted_properties, "sources": len(user_links), "extracted_at": datetime.utcnow(), "cached": False, "stats": extraction_stats}

async def extract_and_geocode_link(extractor: PropertyExtractor, link: Dict) -> tuple:
    link_id = link['id']
    old_properties_key = f"properties:{link_id}"
    try:
        properties = await extractor.extract_properties(link['url'], use_bootstrap=config.ENABLE_BOOTSTRAP_GPT4)
        for prop in properties:
            prop['link_id'] = link_id
        updated_link = link.copy()
        updated_link['last_crawled'] = datetime.utcnow()
        updated_link['properties_count'] = len(properties)
        await set_cached_data(old_properties_key, properties, ttl=3600)
        return properties, updated_link
    except Exception as e:
        logger.error(f"Error extracting and geocoding from {link['url']}: {e}")
        raise e

@app.post("/chat")
async def chat_with_ai(message: ChatMessage, current_user: dict = Depends(get_current_user)):
    user_id = current_user['id']
    if not await check_rate_limit(user_id, "chat"): raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Muitas mensagens. Tente novamente em alguns minutos.")
    try:
        user_properties = await get_cached_data(f"user_properties:{user_id}") or []
        context = create_chat_context(user_properties, message.message)
        if config.OPENAI_API_KEY: response = await generate_ai_response(message.message, context)
        else: response = generate_fallback_response(message.message, user_properties)
        logger.info(f"Chat response generated for user {user_id}")
        return {"response": response, "context_used": len(user_properties), "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Chat error for user {user_id}: {e}")
        return {"response": "Desculpe, ocorreu um erro temporário. Tente novamente em alguns instantes.", "context_used": 0, "timestamp": datetime.utcnow()}

def create_chat_context(properties: List[Dict], user_message: str) -> str:
    if not properties: return "Nenhum imóvel encontrado nos links do usuário."
    total = len(properties)
    types = {}; price_range = {"min": float('inf'), "max": 0}; locations = set()
    for prop in properties:
        prop_type = prop.get('type', 'unknown'); types[prop_type] = types.get(prop_type, 0) + 1
        price = prop.get('price', 0);
        if price > 0: price_range["min"] = min(price_range["min"], price); price_range["max"] = max(price_range["max"], price)
        location = prop.get('location', '')
        if location and location != "Localização não informada": locations.add(location)
    context = f"Total de imóveis encontrados: {total}\n"
    if types: context += "Tipos: " + ", ".join([f"{count} para {tipo}" for tipo, count in types.items()]) + "\n"
    if price_range["min"] != float('inf'): context += f"Preços: R$ {price_range['min']:,.0f} a R$ {price_range['max']:,.0f}\n"
    if locations: context += f"Localizações: {', '.join(list(locations)[:5])}\n"
    return context

async def generate_ai_response(message: str, context: str) -> str:
    try:
        prompt = f"""
Você é o Radar IA, assistente especializado em imóveis. Responda de forma amigável e útil.

Contexto dos imóveis do usuário:
{context}

Pergunta do usuário: {message}

Responda de forma natural e útil, baseando-se nas informações disponíveis.
Mantenha a resposta concisa (máximo 200 palavras).
"""
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise e

def generate_fallback_response(message: str, properties: List[Dict]) -> str:
    message_lower = message.lower()
    if 'quantos' in message_lower or 'total' in message_lower: return f"O radar detectou {len(properties)} imóveis nos seus links monitorados."
    elif 'preço' in message_lower or 'valor' in message_lower:
        if properties:
            prices = [p.get('price', 0) for p in properties if p.get('price', 0) > 0]
            if prices:
                min_price = min(prices); max_price = max(prices); avg_price = sum(prices) / len(prices)
                return f"Os preços variam de R$ {min_price:,.0f} a R$ {max_price:,.0f}, com média de R$ {avg_price:,.0f}."
        return "Não encontrei informações de preço nos imóveis detectados."
    elif 'localização' in message_lower or 'onde' in message_lower:
        if properties:
            locations = [p.get('location', '') for p in properties if p.get('location')]
            unique_locations = list(set(locations))[:5]
            if unique_locations: return f"Os imóveis estão localizados em: {', '.join(unique_locations)}."
        return "Não encontrei informações de localização específicas."
    else: return f"Com base no monitoramento dos seus {len(properties)} imóveis, posso ajudar com informações específicas. O que gostaria de saber?"

@app.post("/bootstrap")
async def bootstrap_extraction(url: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user['id']
    if not await check_rate_limit(user_id, "bootstrap"): raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Muitas requisições de bootstrap. Tente novamente em alguns minutos.")
    if not config.ENABLE_BOOTSTRAP_GPT4 or not config.OPENAI_API_KEY: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Serviço de bootstrap GPT-4 não disponível")
    try:
        async with PropertyExtractor() as extractor:
            properties = await extractor.extract_with_bootstrap(url)
        logger.info(f"Bootstrap extraction completed for {url}: {len(properties)} properties")
        return {"url": url, "properties": properties, "total": len(properties), "method": "bootstrap", "extracted_at": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Bootstrap extraction error for {url}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erro na extração bootstrap")

@app.get("/analytics")
async def get_analytics(current_user: dict = Depends(get_current_user)):
    user_id = current_user['id']
    if not db_pool: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Analytics data requires a database connection.")
    try:
        async with db_pool.acquire() as conn:
            history = await conn.fetch(
                """
                SELECT 
                    h.extracted_at, 
                    h.properties_added, 
                    h.properties_removed, 
                    l.name AS link_name
                FROM extraction_history h
                JOIN links l ON h.link_id = l.id
                WHERE l.user_id = $1
                ORDER BY h.extracted_at DESC
                LIMIT 100
                """,
                user_id
            )
            return {"history": [dict(row) for row in history]}
    except Exception as e:
        logger.error(f"Database error getting analytics data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erro ao carregar dados de analytics.")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(status_code=exc.status_code, content={"error": True, "message": exc.detail, "status_code": exc.status_code, "timestamp": datetime.utcnow().isoformat()})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc} - {request.url}")
    return JSONResponse(status_code=500, content={"error": True, "message": "Erro interno do servidor", "status_code": 500, "timestamp": datetime.utcnow().isoformat()})

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    return response

    app.mount("/static", StaticFiles(directory="static"), name="static")
    @app.get("/favicon.ico")
    async def favicon():
        return FileResponse("static/favicon.ico")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info", reload=config.ENVIRONMENT == "development")