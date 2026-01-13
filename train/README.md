# train

基本的にVibe Coding、動けばよし。

## torch

```
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1
```

```
while true; do python 040_train_weight.py; sleep 10; done
```

## env vars

### PyTorch

```
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### TinyGrad

```
CL=1
```

or

```
AMD=1
```

(needs Resizable Bar)

or https://docs.tinygrad.org/env_vars/
