/**
 * Copie le projet Python dans les ressources du bundle Tauri.
 *
 * On ne duplique rien en VCS : `src-tauri/resources/` est gitignoré et
 * régénéré avant chaque build (`beforeBuildCommand`). Ce qui est embarqué est
 * le strict nécessaire pour que `uv sync --frozen` reconstitue l'environnement :
 * le manifeste, le lock, la version de Python, et le paquet lui-même.
 *
 * Le binaire `uv` est copié en même temps, sous le nom que Tauri exige pour un
 * sidecar : `uv-<triplet cible>`.
 */
import { execFileSync } from "node:child_process";
import { chmodSync, cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const desktop = resolve(here, "..");
const repo = resolve(desktop, "..");
const resources = join(desktop, "src-tauri", "resources", "server");

/**
 * Fichiers et dossiers du dépôt à embarquer, relatifs à sa racine.
 *
 * `README.md` n'est pas de la documentation ici : `pyproject.toml` le déclare en
 * `readme`, et hatchling refuse de construire le wheel s'il est absent.
 * `.python-version` fixe l'interpréteur — sans lui, uv prendrait le plus récent
 * satisfaisant `requires-python`, et le lock résoudrait un autre jeu de paquets.
 */
const PAYLOAD = ["pyproject.toml", "uv.lock", ".python-version", "README.md", "mflux_server"];

function syncPython() {
  rmSync(resources, { recursive: true, force: true });
  mkdirSync(resources, { recursive: true });

  for (const entry of PAYLOAD) {
    const source = join(repo, entry);
    if (!existsSync(source)) {
      throw new Error(`Ressource manquante : ${source}`);
    }
    cpSync(source, join(resources, entry), {
      recursive: true,
      // Ni caches d'octets ni artefacts de test : ils gonflent le bundle et
      // seraient réécrits de toute façon.
      filter: (path) => !/(__pycache__|\.pyc$|\.pytest_cache|\.ruff_cache)/.test(path),
    });
  }
  console.log(`✓ projet Python copié dans ${resources}`);
}

function syncUv() {
  const triple = execFileSync("rustc", ["--print", "host-tuple"], { encoding: "utf8" }).trim();
  const binaries = join(desktop, "src-tauri", "binaries");
  mkdirSync(binaries, { recursive: true });

  let uv;
  try {
    uv = execFileSync("which", ["uv"], { encoding: "utf8" }).trim();
  } catch {
    throw new Error("`uv` introuvable dans le PATH : installe-le avant de builder.");
  }

  // Le suffixe de triplet est imposé par Tauri pour les `externalBin`.
  const target = join(binaries, `uv-${triple}`);
  cpSync(uv, target);
  chmodSync(target, 0o755);
  const version = execFileSync(uv, ["--version"], { encoding: "utf8" }).trim();
  console.log(`✓ sidecar ${version} copié vers uv-${triple}`);
}

syncPython();
syncUv();
