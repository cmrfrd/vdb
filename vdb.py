import pdb
import random
import string
import sys
import traceback
from enum import Enum
from typing import Callable

import numpy as np
import numpy.typing as npt
import tqdm
from pydantic import BaseModel

filename_extension_requirement = ".npz"


class Similarity(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class VDB(BaseModel):
    """A vector database that stores documents and their embeddings."""

    document: list[str]
    embeddings: npt.NDArray[np.float32]
    embedding_function: Callable[[str], npt.NDArray[np.float32]]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def new(
        cls, embedding_function: Callable[[str], npt.NDArray[np.float32]]
    ) -> "VDB":
        """Create a new vector database."""
        return cls(
            document=[],
            embeddings=np.zeros((0, 0), dtype=np.float32),
            embedding_function=embedding_function,
        )

    @classmethod
    def load(
        cls,
        filename: str,
        embedding_function: Callable[[str], npt.NDArray[np.float32]],
    ) -> "VDB":
        """Load a vector database from a file."""
        assert filename.endswith(
            filename_extension_requirement
        ), f"Filename {filename} must end with {filename_extension_requirement}"
        with np.load(filename) as data:
            return cls(
                document=data["document"].tolist(),
                embeddings=data["embeddings"],
                embedding_function=embedding_function,
            )

    def save(self, filename: str) -> None:
        assert filename.endswith(
            filename_extension_requirement
        ), f"Filename {filename} must end with {filename_extension_requirement}"
        np.savez_compressed(
            filename, document=self.document, embeddings=self.embeddings
        )

    def add_document(self, text: str) -> None:
        """Add a document to the vector database."""
        embedding = self.embedding_function(text)
        assert len(embedding.shape) == 1, "Embedding must be a 1D vector"
        self.document.append(text)
        if self.embeddings.size == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.concatenate(
                [self.embeddings, embedding.reshape(1, -1)], axis=0
            )

    def delete_document(self, index: int) -> None:
        """Delete the document at the given index."""
        assert (
            0 <= index < len(self.document)
        ), f"Invalid index {index}, must be between 0 and {len(self.document) - 1}"
        self.document.pop(index)
        self.embeddings = np.delete(self.embeddings, index, axis=0)

    def top_k(
        self,
        query: str,
        k: int = 10,
        similarity: Similarity = Similarity.COSINE,
    ) -> list[str]:
        """Return the top k documents that are most similar to the query."""
        query_embedding = self.embedding_function(query)
        match similarity:
            case Similarity.COSINE:
                similarity_result = np.dot(
                    query_embedding, self.embeddings.T
                ) / (
                    np.linalg.norm(query_embedding)
                    * np.linalg.norm(self.embeddings, axis=1)
                )
            case Similarity.EUCLIDEAN:
                similarity_result = np.linalg.norm(
                    self.embeddings - query_embedding, axis=1
                )
            case _:
                raise ValueError(f"Unknown similarity {similarity}")

        sorted_similarity_indices = np.argsort(similarity_result)[::-1]
        top_k = [self.document[i] for i in sorted_similarity_indices[:k]]
        return top_k


def main() -> None:
    def create_vdb_animal_farm() -> VDB:
        """Create a vector database for the book Animal Farm."""
        filename = "vdb_animal_farm.npz"
        from sentence_transformers import (  # pylint: disable=import-outside-toplevel
            SentenceTransformer,
        )

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        def embedding_func(text: str) -> npt.NDArray[np.float32]:
            return model.encode(text)  # type: ignore

        try:
            vdb = VDB.load(filename, embedding_func)
            return vdb
        except FileNotFoundError:
            vdb = VDB.new(embedding_func)
            with open("./data/animal_farm.txt", "r", encoding="utf8") as f:
                all_text = f.read()
                all_text = all_text.replace("\n", " ")
                all_text = all_text.replace("  ", " ")
                all_text_split = all_text.split(" ")
                # sliding window of 64 words, stride of 32
                for i in tqdm.tqdm(range(0, len(all_text_split) - 64, 32)):
                    text = " ".join(all_text_split[i : i + 64])
                    vdb.add_document(text)
            vdb.save(filename)
            return vdb

    vdb1 = create_vdb_animal_farm()

    def create_vdb_random() -> VDB:
        """Create a vector database with random embeddings."""

        def embedding_func(_: str) -> npt.NDArray[np.float32]:
            """Generate a random embedding."""
            return np.random.rand(1024).astype(np.float32)

        def get_create_vdb() -> VDB:
            filename = "vdb_rand.npz"
            try:
                vdb = VDB.load(filename, embedding_func)
                return vdb
            except FileNotFoundError:
                vdb = VDB.new(embedding_func)
                for _ in tqdm.tqdm(range(10_000)):
                    text = "".join(
                        random.choice(string.ascii_uppercase) for _ in range(64)
                    )
                    vdb.add_document(text)
                vdb.save(filename)
                return vdb

        return get_create_vdb()

    vdb2 = create_vdb_random()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pylint: disable=broad-except
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
