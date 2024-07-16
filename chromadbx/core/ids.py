import uuid
from typing import Generator, Callable, Optional

from chromadb import IDs


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

                >>> ids = IDGenerator(ids_len=10)
                >>> import chromadb
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
