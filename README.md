# Utils to Scrap Japanese Quarterly Financial Reports

### Requirements

- Python: 3.9
  - Poetry
    ```shell
    pip install poetry
    ```
  - Dependencies: refer to pyproject.toml

### Set up Python Virtual Environment

```shell
poetry install
```

### Command Examples

```shell
# scrap a Japanese quarterly financial report (and debug)
poetry run python scripts/scrap_jqfr.py in_file.pdf [--debug debug.pdf]
```
