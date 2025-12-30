#### Particle Jet Tagging with Graph MLP Mixer



#### Installation of dependencies:
```
uv venv jets
source jets/bin/activate
uv pip install torch --no-build-isolation
uv pip install -r pyproject.toml
```
or install dependencies directly using pip
```
pip3 install -r requirements.txt
```

#### Training Procedure:
```
python3 run_training.py configs/toptag_config.py 
```

