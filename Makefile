.PHONY: help install run test eval load smoke lint clean

PY ?= python
export PYTHONPATH := $(CURDIR)/src

help:
	@echo "install  : 安装依赖 (pip install -e . + dev)"
	@echo "run      : 启动 API (http://127.0.0.1:8000/console/)"
	@echo "test     : 运行全部测试"
	@echo "eval     : 运行离线 RAG 评测并写 reports/"
	@echo "load     : 300 请求 / 20 并发的压测"
	@echo "smoke    : 冒烟检查"
	@echo "lint     : ruff 静态检查"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ".[dev]"

run:
	$(PY) -m uvicorn crossmind.api.app:app --host 127.0.0.1 --port 8000 --reload

test:
	$(PY) -m pytest tests -q

eval:
	$(PY) scripts/run_eval.py

load:
	$(PY) scripts/load_test.py --requests 300 --concurrency 20

smoke:
	$(PY) -m crossmind.cli smoke

lint:
	ruff check src tests

clean:
	rm -rf var reports/*.json reports/*.md .pytest_cache .ruff_cache
