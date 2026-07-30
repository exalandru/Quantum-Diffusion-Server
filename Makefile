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
	uv run --project "$(SERVER_DIR)" pytest

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
