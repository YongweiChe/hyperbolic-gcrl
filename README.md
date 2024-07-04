Install requirements:
```
pip install -r requirements.txt
```

Run
```
kernprof -l train.py --config_file configs/hyperbolic_config.yaml
```
with the desired config file to run experiments.

Or run
```
python train.py --config_file configs/hyperbolic_config.yaml
```
for no line profiling.