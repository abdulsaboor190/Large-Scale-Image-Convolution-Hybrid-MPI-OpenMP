CC      = mpicc
CFLAGS  = -O3 -std=gnu11 -Wall -Wextra -fopenmp
LDFLAGS = -fopenmp -lm

BUILD = build
BINS  = $(BUILD)/hello_hybrid $(BUILD)/test_kernels

all: $(BINS)

$(BUILD)/%: src/%.c src/kernels.h | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD):
	mkdir -p $(BUILD)

# Quick end-to-end check of the Phase 0 setup.
check: all
	@echo "--- hybrid smoke test (2 ranks x 2 threads) ---"
	OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 $(BUILD)/hello_hybrid
	@echo "--- kernel check (C vs Python reference) ---"
	python3 scripts/verify_kernels.py

clean:
	rm -rf $(BUILD)

.PHONY: all check clean
