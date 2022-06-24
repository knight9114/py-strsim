# `py-strsim`

The `py-strsim` library is a wrapper for the fabulous
[`strsim`](https://docs.rs/strsim/latest/strsim/) Rust crate. This package
extends the functionality marginally by enabling parallelized versions of each
`strsim` function using [`rayon`](https://docs.rs/rayon/latest/rayon/).


# Installation

It is advised to use a virtual environment for most projects. The instructions
below assume that `python` is the name of your Python-3 interpreter.

```bash
git clone https://github.com/knight9114/py-strsim.git
cd py-strsim
python setup.py install
```


# API

The `py-strsim` package has two parts - `single` and `vectorized`. The `single`
API matches the original `strsim` crate exactly. The `vectorized` versions of
the functions look slightly different:

```python
strsim.vectorized.<function>(n: int, a: str, bs: list[str]) -> list[int] | list[float]:
    ...
```

The first argument, `n`, specifies the number of threads to use during the
computation. Each element in `bs` will be right-compared to the input `a`. The
ordering in the output matches the ordering in the input `bs`.


# Examples

```python
import strsim

assert strsim.single.levenshtein('hello world', 'Hello, World') == 3
assert strsim.single.normalized_levenshtein('hello world', 'Hello, World') == 0.75
...

assert strsim.vectorized.levenshtein(2, 'hello world', ['Hello, World', 'hello world!']) == [3, 1]
...
```


# Credits

My only contribution to this project is writing the Python bindings. All of the credit belongs to
   * [`strsim`](https://github.com/dguo/strsim-rs)
   * [`rayon`](https://github.com/rayon-rs/rayon)
   * [`pyo3`](https://github.com/PyO3/pyo3)