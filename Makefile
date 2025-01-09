PYTHON = python3

GREEN = \033[1;32m
YELLOW = \033[1;33m
RESET = \033[0m

TRAIN_SCRIPT = sources/train.py
TEST_SCRIPT = testing/test.py

train:
	@echo "$(YELLOW)Training model...$(RESET)"
	$(PYTHON) $(TRAIN_SCRIPT)
	@echo "$(GREEN)Successfuly trained model$(RESET)"

test:
	@echo "$(YELLOW)testing model...$(RESET)"
	$(PYTHON) $(TEST_SCRIPT)
	@echo "$(GREEN)Test process is over$(RESET)"

clean:
	rm -rf sources/*__pycache__

.PHONY: train test clean
