from chromadbx.core.queries import (
    eq,
    where,
    ne,
    and_,
    where_document,
    contains,
    lt,
    gt,
    lte,
    gte,
    in_,
    nin,
    not_contains,
    LogicalOperator,
)


def test_eq() -> None:
    assert eq("a", 1).to_dict() == {"a": ["$eq", 1]}


def test_ne() -> None:
    assert ne("a", 1).to_dict() == {"a": ["$ne", 1]}


def test_and() -> None:
    assert and_(eq("a", 1), ne("b", "2")).to_dict() == {
        "$and": [{"a": ["$eq", 1]}, {"b": ["$ne", "2"]}]
    }


def test_lt() -> None:
    assert lt("a", 1).to_dict() == {"a": ["$lt", 1]}


def test_gt() -> None:
    assert gt("a", 1).to_dict() == {"a": ["$gt", 1]}


def test_lte() -> None:
    assert lte("a", 1).to_dict() == {"a": ["$lte", 1]}


def test_gte() -> None:
    assert gte("a", 1).to_dict() == {"a": ["$gte", 1]}


def test_in() -> None:
    assert in_("a", [1, 2, 3]).to_dict() == {"a": ["$in", [1, 2, 3]]}


def test_nin() -> None:
    assert nin("a", [1, 2, 3]).to_dict() == {"a": ["$nin", [1, 2, 3]]}


def test_where_single() -> None:
    assert where(eq("a", 1)) == {"a": ["$eq", 1]}


def test_where_multiple() -> None:
    assert where(and_(eq("a", 1), ne("b", "2"))) == {
        "$and": [{"a": ["$eq", 1]}, {"b": ["$ne", "2"]}]
    }


def test_where_document_contains_single() -> None:
    assert where_document(contains("this is a document")) == {
        "$contains": "this is a document"
    }


def test_where_document_contains_multiple() -> None:
    assert where_document(
        contains("this is a document", "this is another document")
    ) == {
        "$and": [
            {"$contains": "this is a document"},
            {"$contains": "this is another document"},
        ]
    }


def test_where_document_contains_multiple_or() -> None:
    assert where_document(
        contains(
            "this is a document", "this is another document", op=LogicalOperator.OR
        )
    ) == {
        "$or": [
            {"$contains": "this is a document"},
            {"$contains": "this is another document"},
        ]
    }


def test_where_document_not_contains_single() -> None:
    assert where_document(not_contains("this is a document")) == {
        "$not_contains": "this is a document"
    }


def test_where_document_not_contains_multiple() -> None:
    assert where_document(
        not_contains("this is a document", "this is another document")
    ) == {
        "$and": [
            {"$not_contains": "this is a document"},
            {"$not_contains": "this is another document"},
        ]
    }


def test_where_document_not_contains_multiple_or() -> None:
    assert where_document(
        not_contains(
            "this is a document", "this is another document", op=LogicalOperator.OR
        )
    ) == {
        "$or": [
            {"$not_contains": "this is a document"},
            {"$not_contains": "this is another document"},
        ]
    }
