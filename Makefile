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

main:
	$(PYTHON) $(FLAG) $(MODULE).$@

.PHONY: main 