// A stage not ported yet: one card pointing at the classic view, which still has it.
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { legacyUrl } from "@/lib/api";
import type { StageProps } from "./types";

export function StageStub({ stem, what }: StageProps & { what: string }) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <Card className="max-w-sm gap-3 p-5 text-sm">
        <p className="font-medium text-ink">{what[0].toUpperCase() + what.slice(1)} is still in the classic view.</p>
        <Button variant="outline" className="min-h-11 self-start" render={<a href={legacyUrl({ asset: stem })} />} nativeButton={false}>
          Open in classic view
        </Button>
      </Card>
    </div>
  );
}
