import hashlib
import os
import uuid
from functools import partial
from typing import Generator, Callable, Optional

from chromadb import IDs, Documents


def uuid_id_generator() -> Generator[str, None, None]:
    while True:
        yield f"{uuid.uuid4()}"


class IDGenerator(IDs):
    def __init__(
        self,
        ids_len: int,
        generator: Optional[
            Callable[..., Generator[str, None, None]]
        ] = uuid_id_generator,
    ):
        """
        Parameters:
            ids_len (int):  The number of IDs to generate. This must be equal to the number of documents.
            generator (callable):  The function to generate the IDs. The default is uuid_id_generator.

        Example:
            Here's how to use the IDGenerator class:

                >>> import chromadb
                >>> from chromadbx import IDGenerator
                >>> client = chromadb.Client()
                >>> col = client.get_or_create_collection("test")
                >>> my_docs = [f"Document {i}" for i in range(10)]
                >>> col.add(ids=IDGenerator(len(my_docs)), documents=my_docs)

            The above code will generate UUIDs for each document.
        """
        self.ids_len = ids_len
        self._generator = generator()
        self._items = list(next(self._generator) for _ in range(ids_len))

    def __len__(self):
        return self.ids_len

    def __getitem__(self, index):
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def __json__(self):
        return self._items


class UUIDGenerator(IDGenerator):
    def __init__(self, ids_len: int):
        """
        Parameters:
            ids_len (int):  The number of IDs to generate. This must be equal to the number of documents.

        Example:
            Here's how to use the UUIDIDGenerator class:

                >>> import chromadb
                >>> from chromadbx import UUIDGenerator
                >>> client = chromadb.Client()
                >>> col = client.get_or_create_collection("test")
                >>> my_docs = [f"Document {i}" for i in range(10)]
                >>> col.add(ids=UUIDGenerator(len(my_docs)), documents=my_docs)

            The above code will generate UUIDs for each document.
        """
        super().__init__(ids_len=ids_len, generator=uuid_id_generator)


def generate_random_sha256_hash() -> Generator[str, None, None]:
    while True:
        # Generate a random number
        random_data = os.urandom(16)
        # Create a SHA256 hash object
        sha256_hash = hashlib.sha256()
        # Update the hash object with the random data
        sha256_hash.update(random_data)
        # Yield the hexadecimal representation of the hash
        yield sha256_hash.hexdigest()


class RandomSHA256Generator(IDGenerator):
    def __init__(self, ids_len: int):
        """
        Parameters:
            ids_len (int):  The number of IDs to generate. This must be equal to the number of documents.

        Example:
            Here's how to use the SHA256IDGenerator class:

                >>> import chromadb
                >>> from chromadbx import RandomSHA256Generator
                >>> client = chromadb.Client()
                >>> col = client.get_or_create_collection("test")
                >>> my_docs = [f"Document {i}" for i in range(10)]
                >>> col.add(ids=RandomSHA256Generator(len(my_docs)), documents=my_docs)

            The above code will generate SHA256 hashes for each document.
        """
        super().__init__(ids_len=ids_len, generator=generate_random_sha256_hash)


def generate_documents_sha256_hash(documents: Documents) -> Generator[str, None, None]:
    for doc in documents:
        # Create a SHA256 hash object
        sha256_hash = hashlib.sha256()
        # Update the hash object with the random data
        sha256_hash.update(doc.encode("utf-8"))
        # Yield the hexadecimal representation of the hash
        yield sha256_hash.hexdigest()


class DocumentSHA256Generator(IDGenerator):
    def __init__(self, documents: Documents):
        """
        Parameters:
            documents (Documents):  The documents to generate SHA256 hashes for.

        Example:
            Here's how to use the DocumentSHA256IDGenerator class:

                >>> import chromadb
                >>> from chromadbx import DocumentSHA256Generator
                >>> docs = ["Document 1", "Document 2", "Document 3"]
                >>> client = chromadb.Client()
                >>> col = client.get_or_create_collection("test")
                >>> col.add(ids=DocumentSHA256Generator(docs), documents=docs)

            The above code will generate SHA256 hashes for each document.
        """
        super().__init__(
            ids_len=len(documents),
            generator=partial(generate_documents_sha256_hash, documents=documents),
        )


def ulid_generator() -> Generator[str, None, None]:
    try:
        import ulid
    except ImportError:
        raise ValueError(
            "The ulid python package is not installed. Please install it with `pip install ulid-py`"
        )
    while True:
        yield str(ulid.new())


class ULIDGenerator(IDGenerator):
    def __init__(self, ids_len: int):
        """
        Parameters:
            ids_len (int):  The number of IDs to generate. This must be equal to the number of documents.

        Example:
            Here's how to use the ULIDGenerator class:

                >>> import chromadb
                >>> from chromadbx import ULIDGenerator
                >>> client = chromadb.Client()
                >>> col = client.get_or_create_collection("test")
                >>> my_docs = [f"Document {i}" for i in range(10)]
                >>> col.add(ids=ULIDGenerator(len(my_docs)), documents=my_docs)

            The above code will generate ULIDs for each document.
        """
        super().__init__(ids_len=ids_len, generator=ulid_generator)


def nano_id_generator(
    alphabet: Optional[str] = None, size: Optional[int] = None
) -> Generator[str, None, None]:
    try:
        from nanoid import generate
        import nanoid.resources
    except ImportError:
        raise ValueError(
            "The nanoid python package is not installed. Please install it with `pip install nanoid`"
        )
    while True:
        yield str(
            generate(
                alphabet=alphabet or nanoid.resources.alphabet,
                size=size or nanoid.resources.size,
            )
        )


class NanoIDGenerator(IDGenerator):
    def __init__(
        self, ids_len: int, alphabet: Optional[str] = None, size: Optional[int] = None
    ):
        """
        Parameters:
            ids_len (int):  The number of IDs to generate. This must be equal to the number of documents.
            alphabet (str):  The alphabet to use for generating the IDs. The default is `_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`.
            size (int):  The size of the IDs to generate. The default is 21.

        Example:
            Here's how to use the NanoIDGenerator class:

                >>> import chromadb
                >>> from chromadbx import NanoIDGenerator
                >>> client = chromadb.Client()
                >>> col = client.get_or_create_collection("test")
                >>> my_docs = [f"Document {i}" for i in range(10)]
                >>> col.add(ids=NanoIDGenerator(len(my_docs)), documents=my_docs)

            The above code will generate NanoIDs for each document.
        """
        super().__init__(
            ids_len=ids_len,
            generator=partial(nano_id_generator, alphabet=alphabet, size=size),
        )
