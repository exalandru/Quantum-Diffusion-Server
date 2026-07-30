// No console behind the window on Windows — no effect on macOS, but this is the
// canonical shape of a Tauri entry point.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    quantum_diffusion_server_lib::run()
}
