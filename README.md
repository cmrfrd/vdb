# vdb

by: alex@taoa.io

The goal of this project is simple, to install a vector database just copy `vdb.py` into your project + pypi dependencies and start using it.

## Example usage

```python
from vdb import VDB

embedding_function = ... # some function that takes a string and returns a vector

vdb = VDB(embedding_function)
vdb.add_document("hello world!")
vdb.add_document("goodbye world!")
vdb.top_k("hey", k=1) # returns ["hello world!"]
```

## Overview

Vector databases such as [faiss](https://github.com/facebookresearch/faiss), [annoy](https://github.com/spotify/annoy), or [chroma](https://github.com/chroma-core/chroma) are powerful tools. However, they can be difficult to use, require a lot of boilerplate code, and are large potential sources of failure. This project aims to simplify the process of using vector databases by providing a simple copy-paste solution for a performant in memory vector database.

## Developing

`vdb` was developed in vscode in a devcontainer. To install all relevant dependencies, run:

```bash
python3 -m poetry install --no-root --with dev
```
