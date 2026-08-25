/// <reference types="vite/client" />

// Vite's ambient types, which teach TypeScript that an image import yields a
// URL string. Added when the rail's wordmark started importing the app icon
// rather than drawing a coloured square: importing the file is what makes the
// bundler fingerprint it and resolve the `/dashboard/` base, and a bare
// `/app-icon.png` would 404 under that mount.
