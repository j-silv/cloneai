VENV           = .venv
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON))
MODULE 		   = cloneai
DEBUG 		  ?= 0
FLAG		   = -m

ifeq ($(DEBUG), 1)
    FLAG = -m pdb -m
endif


venv:
	rm -rf $(VENV)
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) $(FLAG) $(MODULE).data

clean_data:
	rm -rf data/raw/*/
	rm -f data/raw/**/*.log

.PHONY: venv data clean_data