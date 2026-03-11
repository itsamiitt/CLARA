"""Cache helpers for retrieval results."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

try:
    from redis.asyncio import from_url as redis_from_url
except ImportError:  # pragma: no cover - optional dependency
    redis_from_url = None


GLOBAL_CACHE_NAMESPACE = "__all__"


@dataclass(frozen=True, slots=True)
class CachedScoredMemory:
    """Serializable subset of a scored retrieval hit."""

    memory_id: str
    score: float
    similarity: float
    confidence: float
    recency_score: float
    usage_frequency: float


class MemoryCache:
    """Cache retrieval results in memory or Redis."""

    def __init__(
        self,
        redis_url: str = "memory://",
        *,
        key_prefix: str = "clara",
        default_ttl: int = 3600,
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl
        self._backend = "memory"
        self._client = None
        self._entries: dict[str, tuple[float, str]] = {}
        self._namespaces: dict[str, set[str]] = {}

        if redis_url != "memory://":
            if redis_from_url is None:
                raise ImportError(
                    "The 'redis' package is required to use a Redis-backed MemoryCache."
                )
            self._client = redis_from_url(redis_url, encoding="utf-8", decode_responses=True)
            self._backend = "redis"

    @staticmethod
    def make_query_hash(
        query: str,
        *,
        top_k: int,
        memory_types: Sequence[object] | None = None,
    ) -> str:
        payload = {
            "query": query,
            "top_k": top_k,
            "memory_types": sorted(
                getattr(item, "value", str(item))
                for item in (memory_types or [])
            ),
        }
        return hashlib.blake2b(
            json.dumps(payload, sort_keys=True).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    async def get(
        self,
        user_id: str | None,
        query_hash: str,
    ) -> list[CachedScoredMemory] | None:
        key = self._cache_key(user_id, query_hash)
        payload = await self._get_payload(key)
        if payload is None:
            return None
        raw = json.loads(payload)
        return [CachedScoredMemory(**item) for item in raw]

    async def set(
        self,
        user_id: str | None,
        query_hash: str,
        results: Sequence[CachedScoredMemory],
        *,
        ttl: int | None = None,
    ) -> None:
        key = self._cache_key(user_id, query_hash)
        namespace = self._namespace(user_id)
        payload = json.dumps([asdict(item) for item in results], sort_keys=True)
        ttl = self._default_ttl if ttl is None else ttl

        if self._backend == "redis":
            assert self._client is not None
            await self._client.set(key, payload, ex=ttl)
            await self._client.sadd(self._index_key(namespace), key)
            return

        expires_at = time.monotonic() + max(0, ttl)
        self._entries[key] = (expires_at, payload)
        self._namespaces.setdefault(namespace, set()).add(key)

    async def invalidate(self, user_id: str | None) -> None:
        if user_id is None:
            await self._invalidate_all()
            return

        await self._invalidate_namespace(self._namespace(user_id))
        await self._invalidate_namespace(self._namespace(None))

    async def health(self) -> dict[str, object]:
        if self._backend == "redis":
            assert self._client is not None
            try:
                pong = await self._client.ping()
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                return {"backend": "redis", "ok": False, "error": str(exc)}
            return {"backend": "redis", "ok": bool(pong)}
        return {
            "backend": "memory",
            "ok": True,
            "entries": len(self._entries),
            "namespaces": len(self._namespaces),
        }

    async def close(self) -> None:
        if self._backend == "redis" and self._client is not None:
            await self._client.close()

    def _cache_key(self, user_id: str | None, query_hash: str) -> str:
        return f"{self._key_prefix}:search:{self._namespace(user_id)}:{query_hash}"

    def _index_key(self, namespace: str) -> str:
        return f"{self._key_prefix}:index:{namespace}"

    @staticmethod
    def _namespace(user_id: str | None) -> str:
        return GLOBAL_CACHE_NAMESPACE if user_id is None else user_id

    async def _get_payload(self, key: str) -> str | None:
        if self._backend == "redis":
            assert self._client is not None
            return await self._client.get(key)

        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if expires_at <= time.monotonic():
            self._entries.pop(key, None)
            for keys in self._namespaces.values():
                keys.discard(key)
            return None
        return payload

    async def _invalidate_namespace(self, namespace: str) -> None:
        if self._backend == "redis":
            assert self._client is not None
            index_key = self._index_key(namespace)
            members = list(await self._client.smembers(index_key))
            if members:
                await self._client.delete(*members)
            await self._client.delete(index_key)
            return

        keys = self._namespaces.pop(namespace, set())
        for key in keys:
            self._entries.pop(key, None)

    async def _invalidate_all(self) -> None:
        if self._backend == "redis":
            assert self._client is not None
            search_keys = [key async for key in self._client.scan_iter(f"{self._key_prefix}:search:*")]
            index_keys = [key async for key in self._client.scan_iter(f"{self._key_prefix}:index:*")]
            keys = search_keys + index_keys
            if keys:
                await self._client.delete(*keys)
            return

        self._entries.clear()
        self._namespaces.clear()
