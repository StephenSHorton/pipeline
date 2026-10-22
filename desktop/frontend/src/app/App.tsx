import { useEffect } from "react";
import { HomeScreen } from "@/screens/home/HomeScreen";
import { AssetScreen } from "@/screens/asset/AssetScreen";
import { SettingsScreen } from "@/screens/settings/SettingsScreen";
import { checkFactory, useJobs } from "@/lib/jobs";
import { startUpdateWatch } from "@/screens/settings/update";
import "@/screens/asset/rig/finish"; // rig and render job finishes, on any screen (chain.ts skips them)
import { AssetGate } from "./AssetGate";
import { Overlays } from "./Overlays";
import { TopBar } from "./TopBar";
import { useRoute } from "./router";

export function App() {
  const route = useRoute();
  // At load: /health (the factory line), then reattach to a job the factory is already running.
  useEffect(() => {
    void checkFactory();
    startUpdateWatch();
  }, []);
  // The job card lives on Home. Leaving Home puts a watched job in the background (the chip keeps it), so a
  // job that ends while you look at another asset does not pull you away from it.
  const onHome = route.name === "home";
  useEffect(() => {
    if (!onHome && useJobs.getState().watching) useJobs.getState().setWatching(false);
  }, [onHome]);
  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <TopBar route={route} />
      <main className="relative min-h-0 flex-1">
        {route.name === "asset" ? (
          <AssetGate stem={route.stem}>
            <AssetScreen key={route.stem} stem={route.stem} lod={route.lod} />
          </AssetGate>
        ) : route.name === "settings" ? (
          <SettingsScreen />
        ) : (
          <HomeScreen section={route.section} />
        )}
      </main>
      <Overlays />
    </div>
  );
}
