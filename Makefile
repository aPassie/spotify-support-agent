# All targets run from the repo root. `make reproduce` needs no LLM and no raw data.
PY := .venv/bin/python
LLM := .llm

.PHONY: setup data golden-candidates silver train run judge eval reproduce full quick llm-gen llm-judge llm-stop test clean

setup:            ## create venv (Python 3.12 via uv) and install the package
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

reproduce:        ## headline numbers from committed predictions + judge outputs (<1 min, no LLM)
	$(PY) -m supportagent.evaluate

data:             ## build SpotifyCares subset from data/raw/twcs.csv (downloads it if missing)
	@test -f data/raw/twcs.csv || (mkdir -p data/raw && curl -L -o data/raw/twcs.csv https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv)
	$(PY) -m supportagent.data

golden-candidates:## (re)sample golden candidates; labels live in golden/golden.jsonl and are NOT regenerated
	$(PY) -m supportagent.golden

silver:           ## LLM silver labels for the classifier (needs generator server; ~35 min CPU, cached)
	$(PY) -m supportagent.silver

train:            ## fit retriever + intent classifier (seconds; needs data + silver labels)
	$(PY) -m supportagent.train

run:              ## run agent + baselines on the golden set (needs generator server unless cached)
	$(PY) -m supportagent.run

judge:            ## judge all replies with the judge model (needs judge server unless cached)
	$(PY) -m supportagent.judge_run

eval: reproduce

quick:            ## end-to-end smoke run on 20 golden items with live LLMs (~5 min CPU)
	$(PY) -m supportagent.run --limit 20 && $(PY) -m supportagent.evaluate

full: data train run judge eval   ## everything except silver labelling (use `make silver` first to relabel)

llm-gen:          ## download llama.cpp + generator model and start it on :8080
	bash scripts/setup_llm.sh gen

llm-judge:        ## download judge model and start it on :8081
	bash scripts/setup_llm.sh judge

llm-stop:
	-pkill -f 'llama-server .*--port 808[01]'

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf outputs/models outputs/metrics
