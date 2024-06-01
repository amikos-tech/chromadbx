# collection.get(where={"$and": [{"category": "chroma"}, {"$or": [{"author": "john"}, {"author": "jack"}]}]})
# collection.get(where=Filter.where("category" == "chroma" and ("author" == "john" or "author" == "jack")))
# collection.get(where=where=where(and(eq("category", "chroma"), or(eq("author","john"),eq("author","jack")))))
from typing import Union, Sequence

from chromadb import Where
from chromadb.types import WhereOperator


# where=where(eq("attr", "val"));
# where=where(lte("attr", 10));
# where=where(gte("attr", 10));

class Query:
    def __init__(self, query: dict):
        self.q = query

    def to_dict(self) -> dict:
        return self.q

    def __and__(self, *other: "Query"):
        return and_(*[self.q] + [o.q for o in other])

    def __or__(self, *other: "Query"):
        return or_(*[self.q] + [o.q for o in other])


def where(query: Query) -> Where:
    return query.to_dict()


def eq(attr: str, val: Union[str, int, float, bool]) -> Query:
    return Query({attr: ["$eq", val]})


def lte(attr: str, val: Union[int, float]) -> Query:
    return Query({attr: ["$lte", val]})


def gte(attr: str, val: Union[int, float]) -> Query:
    return Query({attr: ["$gte", val]})


def ne(attr: str, val: Union[str, int, float, bool]) -> Query:
    return Query({attr: ["$ne", val]})


def lt(attr: str, val: Union[int, float]) -> Query:
    return Query({attr: ["$lt", val]})


def gt(attr: str, val: Union[int, float]) -> Query:
    return Query({attr: ["$gt", val]})


def in_(attr: str, val: Sequence[Union[str, int, float, bool]]) -> Query:
    return Query({attr: ["$in", val]})


def nin(attr: str, val: Sequence[Union[str, int, float, bool]]) -> Query:
    return Query({attr: ["$nin", val]})


def and_(*args: Query) -> Query:
    return Query({"$and": [a.to_dict() for a in args]})


def or_(*args: Query) -> Query:
    return Query({"$or": [a.to_dict() for a in args]})


def test_query():
    pass
