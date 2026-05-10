.PHONY: test eval-smoke eval-gold eval-e2e install lint

# Run unit + integration tests (excludes slow model-dependent tests)
test:
	uv run pytest -m "not slow" -q

# Run eval smoke suite (mock predictions, no real models)
eval-smoke:
	uv run pytest -m "eval_smoke" -q

# Run full baseline eval on the gold dataset (requires real ASR models)
eval-gold:
	uv run python eval/runners/run_eval.py \
		--stage baseline \
		--dataset data/eval_datasets/gold_v1/manifest.jsonl \
		--config eval/configs/baseline.yaml \
		--out_root results/

# Run end-to-end eval on the gold dataset (requires all models)
eval-e2e:
	./scripts/eval_e2e.sh

# Install project dependencies
install:
	uv sync

# Run the full test suite
test-all:
	uv run pytest -q
