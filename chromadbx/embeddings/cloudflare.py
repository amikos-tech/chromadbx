# original work on this was done by @mileszim - https://github.com/mileszim/chroma/tree/cloudflare-workers-ai-embedding
import logging
import os
from typing import Optional, Dict, cast

import httpx

from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class CloudflareWorkersAIEmbeddings(EmbeddingFunction[Documents]):  # type: ignore[misc]
    # Follow API Quickstart for Cloudflare Workers AI
    # https://developers.cloudflare.com/workers-ai/
    # Information about the text embedding modules in Google Vertex AI
    # https://developers.cloudflare.com/workers-ai/models/embedding/
    def __init__(
        self,
        model_name: Optional[str] = "@cf/baai/bge-base-en-v1.5",
        *,
        api_token: Optional[str] = os.getenv("CF_API_TOKEN"),
        account_id: Optional[str] = None,
        gateway_endpoint: Optional[
            str
        ] = None,  # use Cloudflare AI Gateway instead of the usual endpoint
        # right now endpoint schema supports up to 100 docs at a time
        # https://developers.cloudflare.com/workers-ai/models/bge-small-en-v1.5/#api-schema (Input JSON Schema)
        max_batch_size: Optional[int] = 100,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Cloudflare Workers AI Embeddings function.

        :param model_name: The name of the model to use. Defaults to "@cf/baai/bge-base-en-v1.5".
        :param api_token: The API token to use. Defaults to the CF_API_TOKEN environment variable.
        :param account_id: The account ID to use.
        :param gateway_endpoint: The gateway URL to use.
        :param max_batch_size: The maximum batch size to use. Defaults to 100.
        :param headers: The headers to use. Defaults to None.
        """
        if not gateway_endpoint and not account_id:
            raise ValueError(
                "Please provide either an account_id or a gateway_endpoint."
            )
        if gateway_endpoint and account_id:
            raise ValueError(
                "Please provide either an account_id or a gateway_endpoint, not both."
            )
        if gateway_endpoint is not None and not gateway_endpoint.endswith("/"):
            gateway_endpoint += "/"
        self._api_url = (
            f"{gateway_endpoint}{model_name}"
            if gateway_endpoint is not None
            else f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
        )
        self._session = httpx.Client()
        self._session.headers.update(headers or {})
        self._session.headers.update({"Authorization": f"Bearer {api_token}"})
        self._max_batch_size = max_batch_size

    def __call__(self, texts: Documents) -> Embeddings:
        # Endpoint accepts up to 100 items at a time. We'll reject anything larger.
        # It would be up to the user to split the input into smaller batches.
        if self._max_batch_size and len(texts) > self._max_batch_size:
            raise ValueError(
                f"Batch too large {len(texts)} > {self._max_batch_size} (maximum batch size)."
            )

        print("URI", self._api_url)

        response = self._session.post(f"{self._api_url}", json={"text": texts})
        response.raise_for_status()
        _json = response.json()
        if "result" in _json and "data" in _json["result"]:
            return cast(Embeddings, _json["result"]["data"])
        else:
            raise ValueError(f"Error calling Cloudflare Workers AI: {response.text}")
