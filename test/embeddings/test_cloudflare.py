import os

import pytest

from chromadbx.embeddings.cloudflare import (
    CloudflareWorkersAIEmbeddings,
)


@pytest.mark.skipif(
    "CF_API_TOKEN" not in os.environ,
    reason="CF_API_TOKEN and CF_ACCOUNT_ID not set, skipping test.",
)
def test_cf_ef_token_and_account() -> None:
    ef = CloudflareWorkersAIEmbeddings(
        api_token=os.environ.get("CF_API_TOKEN", ""),
        account_id=os.environ.get("CF_ACCOUNT_ID"),
    )
    embeddings = ef(["test doc"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


@pytest.mark.skipif(
    "CF_API_TOKEN" not in os.environ,
    reason="CF_API_TOKEN and CF_ACCOUNT_ID not set, skipping test.",
)
def test_cf_ef_gateway() -> None:
    ef = CloudflareWorkersAIEmbeddings(
        api_token=os.environ.get("CF_API_TOKEN", ""),
        gateway_endpoint=os.environ.get("CF_GATEWAY_ENDPOINT"),
    )
    embeddings = ef(["test doc"])
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0


@pytest.mark.skipif(
    "CF_API_TOKEN" not in os.environ,
    reason="CF_API_TOKEN and CF_ACCOUNT_ID not set, skipping test.",
)
def test_cf_ef_large_batch() -> None:
    ef = CloudflareWorkersAIEmbeddings(api_token="dummy", account_id="dummy")
    with pytest.raises(ValueError, match="Batch too large"):
        ef(["test doc"] * 101)


@pytest.mark.skipif(
    "CF_API_TOKEN" not in os.environ,
    reason="CF_API_TOKEN and CF_ACCOUNT_ID not set, skipping test.",
)
def test_cf_ef_missing_account_or_gateway() -> None:
    with pytest.raises(
        ValueError, match="Please provide either an account_id or a gateway_endpoint"
    ):
        CloudflareWorkersAIEmbeddings(api_token="dummy")


@pytest.mark.skipif(
    "CF_API_TOKEN" not in os.environ,
    reason="CF_API_TOKEN and CF_ACCOUNT_ID not set, skipping test.",
)
def test_cf_ef_with_account_or_gateway() -> None:
    with pytest.raises(
        ValueError,
        match="Please provide either an account_id or a gateway_endpoint, not both",
    ):
        CloudflareWorkersAIEmbeddings(
            api_token="dummy", account_id="dummy", gateway_endpoint="dummy"
        )
