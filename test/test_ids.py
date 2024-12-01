import uuid
from functools import partial
from typing import Generator

import chromadb
import pytest
from chromadb import Settings

from chromadbx import (
    IDGenerator,
    NanoIDGenerator,
    ULIDGenerator,
    RandomSHA256Generator,
    DocumentSHA256Generator,
    UUIDGenerator,
)


@pytest.fixture
def client() -> chromadb.Client:
    client = chromadb.Client(settings=Settings(allow_reset=True))
    yield client
    client.reset()


def test_default_generator(client: chromadb.Client) -> None:
    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    col.add(ids=IDGenerator(len(my_docs)), documents=my_docs)
    assert len(col.get()["ids"]) == 10
    assert all(uuid.UUID(_id, version=4) for _id in col.get()["ids"])


def test_uuid_generator(client: chromadb.Client) -> None:
    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    col.add(ids=UUIDGenerator(len(my_docs)), documents=my_docs)
    assert len(col.get()["ids"]) == 10
    assert all(uuid.UUID(_id, version=4) for _id in col.get()["ids"])


def test_custom_generator(client: chromadb.Client) -> None:
    def sequential_generator(start: int = 0) -> Generator[str, None, None]:
        _next = start
        while True:
            yield f"{_next}"
            _next += 1

    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    idgen = IDGenerator(len(my_docs), generator=partial(sequential_generator, start=10))
    col.add(ids=idgen, documents=my_docs)
    assert len(col.get()["ids"]) == 10
    assert col.get()["ids"] == [
        str(i) for i in range(10, 20)
    ]  # this assumes sort order by ID is in effect in Chroma


def test_nano_id_generator(client: chromadb.Client) -> None:
    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    col.add(ids=NanoIDGenerator(len(my_docs)), documents=my_docs)
    assert len(col.get()["ids"]) == 10
    assert all(len(_id) == 21 for _id in col.get()["ids"])


def test_ulid_id_generator(client: chromadb.Client) -> None:
    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    col.add(ids=ULIDGenerator(len(my_docs)), documents=my_docs)
    assert len(col.get()["ids"]) == 10
    import ulid

    assert all(isinstance(ulid.parse(_id), ulid.ULID) for _id in col.get()["ids"])


def test_random_sha256_id_generator(client: chromadb.Client) -> None:
    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    col.add(ids=RandomSHA256Generator(len(my_docs)), documents=my_docs)
    assert len(col.get()["ids"]) == 10
    assert all(len(_id) == 64 for _id in col.get()["ids"])


def test_document_sha256_id_generator(client: chromadb.Client) -> None:
    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    col.add(ids=DocumentSHA256Generator(documents=my_docs), documents=my_docs)
    assert len(col.get()["ids"]) == 10
    assert all(len(_id) == 64 for _id in col.get()["ids"])
    assert all(
        _id in col.get()["ids"]
        for _id in list(DocumentSHA256Generator(documents=my_docs))
    )
