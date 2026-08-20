import React from "react";
import { createRoot } from "react-dom/client";

import { PlaygroundApp } from "./PlaygroundApp";
import "../styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from playground.html");

createRoot(root).render(
  <React.StrictMode>
    <PlaygroundApp />
  </React.StrictMode>,
);
