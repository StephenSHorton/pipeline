import "@fontsource-variable/inter";
import "mokei/theme/yardline"; // registers the theme
import { applyTheme, setDefaultThemeId } from "mokei/theme";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { TooltipProvider } from "@/components/ui/tooltip";
import { App } from "./app/App";
import "./index.css";

setDefaultThemeId("yardline");
applyTheme("yardline"); // index.html already sets data-theme="yardline" (no flash); this runs the palette check

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <TooltipProvider>
      <App />
    </TooltipProvider>
  </StrictMode>,
);
