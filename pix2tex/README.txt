
pytorch和torchtext不匹配，会报错：
OSError: /mnt/AI/Latex/EasyLatexServer/venv/lib/python3.12/site-packages/torchtext/lib/libtorchtext.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs


解决办法：
pip uninstall torch
pip uninstall torchtext
pip install torch==2.3.0
pip install torchtext==0.18
