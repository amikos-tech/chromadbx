from pydantic import BaseModel


class Record(BaseModel):
    id: str


class Collection(BaseModel):
    pass
