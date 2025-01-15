from typing import Optional, cast

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


class SpacyEmbeddingFunction(EmbeddingFunction[Documents]):  # type: ignore[misc]
    """
    SpacyEmbeddingFunction is an embedding function that uses the spacy library to get embeddings for a list of texts. See https://spacy.io/usage/models for more information.
    """

    def __init__(self, model_name: Optional[str] = "en_core_web_lg"):
        """
        Initialize the SpacyEmbeddingFunction.
        The default model is "en_core_web_lg" which is a large model that optimizes accuracy and has embeddings in-built.

        Args:
            model_name (str): The name of the spacy model to use.
            default: "en_core_web_lg"

        """
        try:
            import spacy
        except ImportError:
            raise ValueError(
                "The spacy python package is not installed. Please install it with `pip install spacy`"
            )
        self._model_name = model_name

        try:
            # disable ner, tagger, parser, attribute_ruler, lemmatizer to speed up the model
            self._nlp = spacy.load(
                str(self._model_name),
                disable=["ner", "tagger", "parser", "attribute_ruler", "lemmatizer"],
            )
        except OSError:
            raise ValueError(
                f"""spacy model '{self._model_name}' are not downloaded yet, please download them using `python -m spacy download {self._model_name}`, Please checkout
                for the list of models from: https://spacy.io/usage/models."""
            )

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.

        Args:
            input (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: The embeddings for the texts.
        Example:
            >>> spacy_fn = SpacyEmbeddingFunction(model_name="en_core_web_lg")
            >>> input = ["Hello, world!", "How are you?"]
            >>> embeddings = spacy_fn(input)
        """
        embeddings = []
        for em in self._nlp.pipe(input):
            embeddings.append(em.vector.astype("float"))

        return cast(Embeddings, embeddings)
