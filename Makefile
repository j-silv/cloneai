VENV           = venv
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON))
MODULE 		   = cloneai
DEBUG 		   ?= 0
FLAG		   = -m

ifeq ($(DEBUG), 1)
    FLAG = -m pdb -m
endif

venv:
	rm -rf $(VENV)
	$(SYSTEM_PYTHON) -m $@ $(VENV)
	$(VENV_PYTHON) -m pip install -r requirements.txt

get_weights:
	$(PYTHON) $(FLAG) $(MODULE).$@

main:
	$(PYTHON) $(FLAG) $(MODULE).$@

extract:
	$(PYTHON) $(FLAG) $(MODULE).data.$@

split:
	$(PYTHON) $(FLAG) $(MODULE).data.$@

transcribe:
	$(PYTHON) $(FLAG) $(MODULE).data.$@

voice:
	$(PYTHON) $(FLAG) $(MODULE).bot.$@

tts:
	$(PYTHON) $(FLAG) $(MODULE).$@.tacotron2

test:
	$(PYTHON) $(FLAG) $(MODULE).$@

clean_raw:
	rm -rf data/raw/*/
	rm -f data/raw/**/*.log

clean_processed:
	rm -rf data/processed/*

clean_extracted:
	rm -rf data/extracted

.PHONY: venv extract split transcribe test clean_raw clean_processed