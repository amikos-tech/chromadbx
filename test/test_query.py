from chromadbx.core.queries import eq, where, ne, and_


def test_where():
    print(where(and_(eq("a", 1), ne("b", "2"))))
