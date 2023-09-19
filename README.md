# intelligence-layer

```
conda create -n intelligence-layer python=3.10 -y
conda activate intelligence-layer
pip install -e .[test]
pre-commit install
pre-commit run -a
mypy src
mypy tests
pytest
```
