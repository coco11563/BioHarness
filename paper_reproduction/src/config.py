"""Unified service configuration for the paper-reproduction code.

Released version: every endpoint and credential comes from an environment variable.
The defaults are neutral localhost placeholders; they are NOT the values of the
research deployment. The class and attribute names are unchanged, so the rest of
the research code reads this module exactly as before.

Environment variables (comma-separated lists where plural):

  XC_LLM_SERVERS        OpenAI-compatible chat endpoints     (default http://127.0.0.1:8005/v1)
  XC_LLM_MODEL          served model name                    (default Qwen/Qwen3.5-35B-A3B)
  XC_LLM_KEY            API key                              (default EMPTY)
  XC_RERANK_SERVERS     reranker endpoints (vLLM, OpenAI API) (default http://127.0.0.1:8015/v1,http://127.0.0.1:8016/v1)
  XC_RERANK_MODEL       served reranker name                 (default Qwen/Qwen3-Reranker-8B)
  XC_EMBED_SERVERS      embedding endpoints                  (default http://127.0.0.1:8026/v1,http://127.0.0.1:8027/v1)
  XC_EMBED_MODEL        served embedding name                (default Qwen/Qwen3-Embedding-0.6B)
  XC_PG_HOST            PostgreSQL host                      (default 127.0.0.1)
  XC_PG_PORTS           PostgreSQL ports, round-robin        (default 5432)
  XC_PG_USER            PostgreSQL user                      (default postgres)
  XC_PG_PASSWORD        PostgreSQL password                  (default change-me)
  XC_PG_PUBMED_DB       PubMed abstract database             (default paper-graph-pubmed)
  XC_PG_PAPERGRAPH_DB   PMC full-text database               (default papergraph)
  XC_QDRANT_URLS        Qdrant REST URLs, round-robin        (default http://127.0.0.1:6333)
  XC_QDRANT_NO_PROBE=1  skip the one-time liveness probe of the Qdrant URLs
"""

from dataclasses import dataclass, field
import itertools
import os


def _env_list(name: str, default: str) -> list[str]:
    return [u.strip() for u in os.environ.get(name, default).split(",") if u.strip()]


@dataclass
class LLMEndpointConfig:
    """Configuration for a single LLM endpoint."""
    url: str
    model: str
    weight: int = 1  # Higher weight = more requests


@dataclass
class LLMConfig:
    """LLM server configuration with load balancing support."""

    endpoints: list[LLMEndpointConfig] = field(default_factory=list)

    api_key: str = "EMPTY"
    max_tokens: int = 2048
    temperature: float = 0.1

    # Legacy compatibility: list of server URLs
    @property
    def servers(self) -> list[str]:
        """Get list of server URLs (legacy compatibility)."""
        return [ep.url for ep in self.endpoints]

    @property
    def model(self) -> str:
        """Get primary model name (legacy compatibility)."""
        return self.endpoints[0].model if self.endpoints else "qwen"

    def __post_init__(self):
        if not self.endpoints:
            m = os.environ.get("XC_LLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
            self.endpoints = [LLMEndpointConfig(url=u, model=m, weight=1)
                              for u in _env_list("XC_LLM_SERVERS", "http://127.0.0.1:8005/v1")]
        if os.environ.get("XC_LLM_KEY"):
            self.api_key = os.environ["XC_LLM_KEY"]
        self._endpoint_cycle = itertools.cycle(self.endpoints)

    def get_server(self) -> str:
        """Get next server URL in round-robin fashion (legacy)."""
        return next(self._endpoint_cycle).url

    def get_endpoint(self) -> LLMEndpointConfig:
        """Get next endpoint in round-robin fashion."""
        return next(self._endpoint_cycle)

    @property
    def primary(self) -> str:
        """Get primary server URL."""
        return self.endpoints[0].url if self.endpoints else ""


@dataclass
class RerankConfig:
    """Rerank server configuration (Qwen3-Reranker-8B served by vLLM)."""
    servers: list[str] = field(default_factory=lambda: _env_list(
        "XC_RERANK_SERVERS", "http://127.0.0.1:8015/v1,http://127.0.0.1:8016/v1"))
    model: str = field(default_factory=lambda: os.environ.get(
        "XC_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"))
    api_key: str = "EMPTY"
    max_tokens: int = 32768
    temperature: float = 0.1

    def __post_init__(self):
        self._server_cycle = itertools.cycle(self.servers)

    def get_server(self) -> str:
        """Get next server in round-robin fashion."""
        return next(self._server_cycle)

    @property
    def primary(self) -> str:
        """Get primary server URL."""
        return self.servers[0] if self.servers else ""


@dataclass
class EmbeddingConfig:
    """Embedding server configuration (Qwen3-Embedding-0.6B served by vLLM)."""

    servers: list[str] = field(default_factory=lambda: _env_list(
        "XC_EMBED_SERVERS", "http://127.0.0.1:8026/v1,http://127.0.0.1:8027/v1"))

    model: str = field(default_factory=lambda: os.environ.get(
        "XC_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    api_key: str = "EMPTY"
    dimension: int = 1024

    def __post_init__(self):
        self._server_cycle = itertools.cycle(self.servers)

    def get_server(self) -> str:
        """Get next server in round-robin fashion."""
        return next(self._server_cycle)

    def get_model(self) -> str:
        """Get model name."""
        return self.model


@dataclass
class PostgresConfig:
    """PostgreSQL database configuration.

    Several ports may point at the same database; connections are spread over them
    round-robin.
    """

    ports: list[int] = field(default_factory=lambda: [
        int(p) for p in _env_list("XC_PG_PORTS", "5432")])
    host: str = field(default_factory=lambda: os.environ.get("XC_PG_HOST", "127.0.0.1"))

    # Database names
    pubmed_db: str = field(default_factory=lambda: os.environ.get(
        "XC_PG_PUBMED_DB", "paper-graph-pubmed"))
    papergraph_db: str = field(default_factory=lambda: os.environ.get(
        "XC_PG_PAPERGRAPH_DB", "papergraph"))
    user: str = field(default_factory=lambda: os.environ.get("XC_PG_USER", "postgres"))
    password: str = field(default_factory=lambda: os.environ.get("XC_PG_PASSWORD", "change-me"))

    def __post_init__(self):
        self._port_cycle = itertools.cycle(self.ports)

    def _build_url(self, db: str, port: int) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{port}/{db}"

    @property
    def pubmed_url(self) -> str:
        """Get next PubMed URL in round-robin."""
        return self._build_url(self.pubmed_db, next(self._port_cycle))

    @property
    def papergraph_url(self) -> str:
        """Get next PaperGraph URL in round-robin."""
        return self._build_url(self.papergraph_db, next(self._port_cycle))

    @property
    def pubmed_urls(self) -> list[str]:
        """All PubMed URLs (one per port)."""
        return [self._build_url(self.pubmed_db, p) for p in self.ports]

    @property
    def papergraph_urls(self) -> list[str]:
        """All PaperGraph URLs (one per port)."""
        return [self._build_url(self.papergraph_db, p) for p in self.ports]

    # Legacy compatibility
    @property
    def pubmed_async_url(self) -> str:
        return self.pubmed_url.replace("postgresql://", "postgresql+asyncpg://")

    @property
    def papergraph_async_url(self) -> str:
        return self.papergraph_url.replace("postgresql://", "postgresql+asyncpg://")


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration.

    Several URLs may point at the same Qdrant service; clients are spread over them
    round-robin.
    """

    urls: list[str] = field(default_factory=lambda: _env_list(
        "XC_QDRANT_URLS", "http://127.0.0.1:6333"))

    # Legacy single-URL access
    @property
    def url(self) -> str:
        """Next URL in round-robin, restricted to URLs that actually answer.

        When one of several URLs is down, plain round-robin hands a dead URL to every
        caller that keeps a single client, and callers without failover then fail
        for the life of the process. The probe runs once per process;
        XC_QDRANT_NO_PROBE=1 skips it.
        """
        if self._live_cycle is None:
            self._live_cycle = itertools.cycle(self._probe_live_urls())
        return next(self._live_cycle)

    def _probe_live_urls(self) -> list[str]:
        if os.environ.get("XC_QDRANT_NO_PROBE") == "1":
            return list(self.urls)
        import socket
        from urllib.parse import urlparse

        live = []
        for u in self.urls:
            p = urlparse(u)
            try:
                with socket.create_connection((p.hostname, p.port or 6333), timeout=2):
                    live.append(u)
            except OSError:
                pass
        if not live:  # nothing answered: keep the declared list rather than break hard
            return list(self.urls)
        return live

    def __post_init__(self):
        self._url_cycle = itertools.cycle(self.urls)
        self._live_cycle = None

    # Collections
    paper_collection: str = "paper-full"        # 27.3M PubMed abstracts
    mesh_collection: str = "mesh-term-only"     # 30,956 MeSH terms
    mesh_full_collection: str = "mesh-terms"    # MeSH with scope notes
    chunks_collection: str = "chunks"           # PMC full-text chunks

    # Suppress version warnings
    check_compatibility: bool = False


@dataclass
class Config:
    """Main configuration combining all components."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


# Default singleton instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set a custom configuration instance."""
    global _config
    _config = config


# Quick access functions
def get_llm_server() -> str:
    """Get next LLM server URL (round-robin)."""
    return get_config().llm.get_server()


def get_embed_server() -> str:
    """Get next embedding server URL (round-robin)."""
    return get_config().embedding.get_server()


def get_embed_model() -> str:
    """Get embedding model name."""
    return get_config().embedding.model


def get_rerank_server() -> str:
    """Get next rerank server URL (round-robin)."""
    return get_config().rerank.get_server()


if __name__ == "__main__":
    # Print the resolved configuration (passwords masked).
    config = get_config()
    print("LLM      :", config.llm.servers, "model =", config.llm.model)
    print("Rerank   :", config.rerank.servers, "model =", config.rerank.model)
    print("Embedding:", config.embedding.servers, "model =", config.embedding.model)
    pg = config.postgres
    print("Postgres :", f"{pg.user}:***@{pg.host}:{pg.ports}", pg.pubmed_db, pg.papergraph_db)
    print("Qdrant   :", config.qdrant.urls, "collection =", config.qdrant.paper_collection)
