from typing import Dict, Union, Optional, Iterable

from chromadb import Where, WhereDocument
from chromadb.api.types import Embedding, ID, OneOrMany, Include
from pydantic import BaseModel


class Record(BaseModel):
    id: ID
    embedding: Embedding
    metadata: Optional[Dict[str, Union[str, int, float, bool]]]
    document: Optional[str] = None
    uri: Optional[str] = None



class Collection(BaseModel):
    def add(self, *records: Record) -> None:
        pass

    def get(
            self,
            ids: Optional[OneOrMany[ID]] = None,
            where: Optional[Where] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[WhereDocument] = None,
            include: Include = ["metadatas", "documents"],
    ) -> None:
        pass

    def __iter__(self):
        """Allows iteration over the collection's records."""
        return self



class RecordSet:
    def __init__(self, collection: Collection, records: Iterable[Record]) -> None:
        self.records = records


