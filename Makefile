# Convenience targets
.PHONY: demo env data train predict eval

env:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	. .venv/bin/activate; python src/build_dataset.py

train:
	. .venv/bin/activate; python src/train_start_detector.py --train_glob "data/build/*/O0/*_sym" --model_path models/start_detector.joblib

predict:
	. .venv/bin/activate; python src/predict_starts.py --bin "data/build/linux/O3/hello_stripped" --model_path models/start_detector.joblib --out functions_pred.json

eval:
	. .venv/bin/activate; python src/eval_starts.py --pred functions_pred.json --truth_glob "data/labels/*/O3/hello_sym.functions_truth.json" --tolerance 8
