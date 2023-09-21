# intelligence-layer

```
conda create -n intelligence-layer python=3.10 -y
conda activate intelligence-layer
pip install -e .[test]
pre-commit install
pre-commit run -a

mypy
pytest
```


Next steps:
- [X] Typing of generic classes
- [ ] Put classify in
- [ ] Generate + classify the output
- [ ] UI for classify with audit trail, examples, ...
