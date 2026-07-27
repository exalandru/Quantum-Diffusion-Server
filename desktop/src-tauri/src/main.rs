// Pas de console derrière la fenêtre sur Windows — sans effet sur macOS, mais
// c'est la forme canonique d'un point d'entrée Tauri.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    quantum_diffusion_server_lib::run()
}
