import uuid
from functools import partial
from typing import Generator

import chromadb
import pytest
from chromadb import Settings

from chromadbx import IDGenerator


@pytest.fixture
def client() -> chromadb.Client:
    client = chromadb.Client(settings=Settings(allow_reset=True))
    yield client
    client.reset()


def test_default_generator(client) -> None:
    col = client.get_or_create_collection("test")
    my_docs = [f"Document {_}" for _ in range(10)]
    col.add(ids=IDGenerator(len(my_docs)), documents=my_docs)
    assert len(col.get()["ids"]) == 10
    for _id in col.get()["ids"]:
        assert uuid.UUID(_id, version=4)


def test_custom_generator(client) -> None:
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
    assert col.get()["ids"] == [str(i) for i in range(10, 20)]
