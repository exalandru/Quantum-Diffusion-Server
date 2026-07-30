# Quantum Diffusion Server

This repository contains two applications:

- [`src/server`](src/server/README.md): the Python API server exposing mflux
  through an OpenAI Images-compatible API.
- [`src/desktop`](src/desktop/README.md): the Tauri and React macOS control
  panel.

## Repository layout

```text
src/
├── server/       Python package, tests and configuration
└── desktop/      React frontend and Tauri application
build/            disposable compiler output and bundle staging
dist/             distributable wheels, source archives, .app and .dmg files
```

Both `build/` and `dist/` are generated and ignored by Git.

## Common commands

```sh
make install
make test
make lint
make dev-server
make dev-desktop
make build
make clean
```

Run `make help` for the complete list.
