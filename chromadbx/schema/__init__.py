import chromadb
def collection_schema_builder(client:chromadb.ClientAPI, collection_name):
    """
    Builds a collection Schema which provides insights into the information stored in the collection.
    Should allow for an easy self-query by LLMs.

    :param client:
    :param collection_name:
    :return:
    """
    collection = client.get_collection(collection_name)
    if collection is None:
        raise ValueError(f"Collection {collection_name} not found")

    return collection