ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SERVER_DIR := $(ROOT)/src/server
DESKTOP_DIR := $(ROOT)/src/desktop
BUILD_DIR := $(ROOT)/build
DIST_DIR := $(ROOT)/dist

export UV_PROJECT_ENVIRONMENT := $(ROOT)/.venv

.PHONY: help install install-server install-desktop dev-server dev-desktop \
	test lint build build-server build-desktop clean

help:
	@printf '%s\n' \
		'make install        Install server and desktop dependencies' \
		'make dev-server     Run the Python API server' \
		'make dev-desktop    Run the Tauri desktop application' \
		'make test           Run the Python test suite' \
		'make lint           Run Python lint and TypeScript checks' \
		'make build          Build all distributable artifacts' \
		'make build-server   Build Python wheel and source archive' \
		'make build-desktop  Build QDS.app and QDS.dmg' \
		'make clean          Remove build/ and dist/'

install: install-server install-desktop

install-server:
	uv sync --project "$(SERVER_DIR)"

install-desktop:
	npm --prefix "$(DESKTOP_DIR)" install

dev-server:
	uv run --project "$(SERVER_DIR)" mflux-server

dev-desktop:
	npm --prefix "$(DESKTOP_DIR)" run app:dev

test:
	# `python -m pytest`, not the `pytest` console script. The script is generated
	# at install time with an absolute shebang, so a shared environment that was
	# built before the repository moved still points its scripts at the old path
	# and `exec` fails with a bare "Failed to spawn: pytest". Running the module
	# goes through the interpreter uv resolves, which is a symlink and survives
	# the move. (`ruff` above is unaffected only because it is a native binary
	# with no shebang at all.)
	uv run --project "$(SERVER_DIR)" python -m pytest "$(SERVER_DIR)/tests"
	# Same target directory as `app:build`, so the tests share its cache and
	# nothing lands inside `src-tauri/`, which `clean` does not reach.
	cd "$(DESKTOP_DIR)/src-tauri" && CARGO_TARGET_DIR="$(BUILD_DIR)/desktop/tauri" cargo test
	npm --prefix "$(DESKTOP_DIR)" test

lint:
	uv run --project "$(SERVER_DIR)" ruff check .
	npm --prefix "$(DESKTOP_DIR)" run typecheck

build: build-server build-desktop

build-server:
	mkdir -p "$(DIST_DIR)/server"
	uv build "$(SERVER_DIR)" --out-dir "$(DIST_DIR)/server"

build-desktop:
	npm --prefix "$(DESKTOP_DIR)" run app:build

clean:
	rm -rf "$(BUILD_DIR)" "$(DIST_DIR)"
