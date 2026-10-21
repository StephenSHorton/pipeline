// ?asset=<stem> before the asset screen: a stem the factory does not list gets a plain "no asset named" card with
// Back to Home. If /assets itself fails, the asset is shown anyway.
import { useEffect, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { listAssets } from "@/lib/api";
import { navigate } from "./router";

type Gate = "loading" | "found" | "missing";

export function AssetGate({ stem, children }: { stem: string; children: ReactNode }) {
  const [gate, setGate] = useState<Gate>("loading");
  useEffect(() => {
    let live = true;
    setGate("loading");
    listAssets()
      .then((r) => {
        if (!live) return;
        const a = r.assets.find((x) => x.id === stem);
        setGate(a ? "found" : "missing");
      })
      .catch(() => live && setGate("found"));
    return () => {
      live = false;
    };
  }, [stem]);
  if (gate === "found") return <>{children}</>;
  if (gate === "loading") {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <Card size="hud" role="status" className="max-w-[420px] rounded-[14px] px-4 py-2.5 text-center text-ink">
          Loading…
        </Card>
      </div>
    );
  }
  return (
    <div className="flex h-full items-center justify-center p-4">
      <Card className="max-w-sm gap-3 p-5 text-sm">
        <p className="font-medium">There is no asset named “{stem}”.</p>
        <p className="text-ink-3">It may have been renamed or deleted.</p>
        <Button className="min-h-11 self-start" onClick={() => navigate({ name: "home" }, { replace: true })}>Back to Home</Button>
      </Card>
    </div>
  );
}
