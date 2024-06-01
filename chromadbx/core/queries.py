from enum import Enum
from typing import Union, Sequence, Optional, Any, Dict

from chromadb import Where, WhereDocument


class Query:
    def __init__(self, query: Dict[str, Any]):
        self.q = query

    def to_dict(self) -> Dict[str, Any]:
        return self.q

    def __and__(self, *other: "Query") -> "Query":
        return and_(self, *other)

    def __or__(self, *other: "Query") -> "Query":
        return or_(self, *other)


def where(query: Query) -> Where:
    return query.to_dict()


def where_document(query: Query) -> WhereDocument:
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


class LogicalOperator(str, Enum):
    AND = "$and"
    OR = "$or"


def contains(*val: str, op: Optional[LogicalOperator] = LogicalOperator.AND) -> Query:
    if len(val) == 1:
        return Query({"$contains": val[0]})
    if op is LogicalOperator.AND:
        return Query({"$and": [{"$contains": v} for v in val]})
    return Query({"$or": [{"$contains": v} for v in val]})


def not_contains(
    *val: str, op: Optional[LogicalOperator] = LogicalOperator.AND
) -> Query:
    if len(val) == 1:
        return Query({"$not_contains": val[0]})
    if op is LogicalOperator.AND:
        return Query({"$and": [{"$not_contains": v} for v in val]})
    return Query({"$or": [{"$not_contains": v} for v in val]})
