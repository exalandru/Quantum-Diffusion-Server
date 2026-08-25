ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SERVER_DIR := $(ROOT)/src/server
DASHBOARD_DIR := $(ROOT)/src/dashboard
MENUBAR_DIR := $(ROOT)/src/menubar
BUILD_DIR := $(ROOT)/build
DIST_DIR := $(ROOT)/dist

export UV_PROJECT_ENVIRONMENT := $(ROOT)/.venv

.PHONY: help install install-server install-dashboard dev-server dev-dashboard \
	test test-server test-dashboard test-app lint build build-server build-dashboard \
	build-app build-dmg clean-build clean

help:
	@printf '%s\n' \
		'make install          Install server and dashboard dependencies' \
		'make dev-server       Run the Python API server' \
		'make dev-dashboard    Run the dashboard against a running server' \
		'make test             Run the test suites' \
		'make lint             Run Python lint and TypeScript checks' \
		'make build            Build the wheel, QDS.app and the DMG' \
		'make build-dashboard  Build the web dashboard into the Python package' \
		'make build-server     Build the wheel, dashboard included' \
		'make build-app        Build QDS.app, wheel and uv included' \
		'make build-dmg        Package QDS.app into dist/app/QDS-<version>.dmg' \
		'make clean            Remove build/ and dist/'

install: install-server install-dashboard

install-server:
	uv sync --project "$(SERVER_DIR)"

install-dashboard:
	npm --prefix "$(DASHBOARD_DIR)" install

dev-server:
	uv run --project "$(SERVER_DIR)" qds serve

# Vite serves the page and proxies the API to a server you start separately —
# `make dev-server` in another terminal. Same-origin in development as in
# production, which is what the control plane requires.
dev-dashboard:
	npm --prefix "$(DASHBOARD_DIR)" run dev

test: test-server test-dashboard test-app

test-server:
	# `python -m pytest`, not the `pytest` console script. The script is generated
	# at install time with an absolute shebang, so a shared environment that was
	# built before the repository moved still points its scripts at the old path
	# and `exec` fails with a bare "Failed to spawn: pytest". Running the module
	# goes through the interpreter uv resolves, which is a symlink and survives
	# the move. (`ruff` above is unaffected only because it is a native binary
	# with no shebang at all.)
	uv run --project "$(SERVER_DIR)" python -m pytest "$(SERVER_DIR)/tests"

test-dashboard:
	npm --prefix "$(DASHBOARD_DIR)" test

test-app:
	swift test --package-path "$(MENUBAR_DIR)"

lint:
	uv run --project "$(SERVER_DIR)" ruff check "$(SERVER_DIR)"
	npm --prefix "$(DASHBOARD_DIR)" run typecheck

build: build-server build-app build-dmg clean-build

build-dashboard:
	npm --prefix "$(DASHBOARD_DIR)" run build

# The wheel carries `qds/_dashboard`, so building the front end is not an
# optional preceding step — it is part of producing a correct wheel. The guard
# turns "shipped a server with no interface" from something discovered after
# installing into something that fails here.
build-server: build-dashboard
	@test -f "$(SERVER_DIR)/qds/_dashboard/index.html" \
		|| { echo "qds/_dashboard/index.html is missing: the wheel would ship without a dashboard."; exit 1; }
	mkdir -p "$(DIST_DIR)/server"
	uv build "$(SERVER_DIR)" --out-dir "$(DIST_DIR)/server"

# The app bundles the wheel, so it cannot be built before one exists.
build-app: build-server
	"$(ROOT)/scripts/bundle-menubar.sh"

# What the Tauri bundler used to produce, and what v2 lost with it: the app on
# its own is not something to hand someone.
build-dmg: build-app
	"$(ROOT)/scripts/make-dmg.sh"

clean-build:
	rm -rf "$(BUILD_DIR)" "$(MENUBAR_DIR)/.build"

clean:
	rm -rf "$(BUILD_DIR)" "$(DIST_DIR)" "$(MENUBAR_DIR)/.build"
