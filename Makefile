VENV           = venv
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON))
MODULE 		   = cloneai
WAVERNN 	   = $(MODULE)/wavernn
DEBUG 		   ?= 0
FLAG		   = -m

ifeq ($(DEBUG), 1)
    FLAG = -m pdb -m
endif

main:
	$(PYTHON) $(FLAG) $(MODULE).$@


download_ljspeech:
	wavernn dataset download ljspeech --destination $(WAVERNN)/ljspeech
wavernn_ljspeech:
	wavernn train --config $(WAVERNN)/config/wavernn.yaml --path $(WAVERNN)/runs/my-model --data ./$(WAVERNN)/ljspeech

wavernn_train:
	wavernn train --config $(WAVERNN)/config/wavernn.yaml --path $(WAVERNN)/runs/my-model --data ./data/processed/1-scott --test-every 100

wavernn_infer:
	wavernn infer \
		--path $(WAVERNN)runs/my-model \
		--input ./data/processed/1-scott/craig_ZYNzcjw2Jx7d_2025-3-21_21-0-25_0_52.wav  \
		--output resynthesized-scott.wav

wavernn_export:
	wavernn export --path $(WAVERNN)/runs/my-model --output $(WAVERNN)/runs/my-exported-model.jit


.PHONY: main wavernn_train wavernn_ljspeech 