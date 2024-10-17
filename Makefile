set_env:
	conda create -n gVit
	python3 -m pip install -r requirements.txt

train:
	python3 run.py train $(RUN_ARGS)

validate:
	python3 run.py validate $(RUN_ARGS)

all:
	train 
	validate

