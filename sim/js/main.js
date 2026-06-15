/** Entry point for the 3D joint simulator. */

import { bootstrap } from "./bootstrap.js";

document.getElementById("reloadCfg").addEventListener("click", () => location.reload());
bootstrap().catch((err) => {
  console.error("bootstrap failed:", err);
  const loading = document.getElementById("loading");
  loading.textContent = String(err?.message || err);
  loading.style.display = "grid";
});
