// The Cleanup panel's one factory call, POST /clean (legacy Floaters and Carve handlers, index.html 5319-5370),
// and its busy flag. /clean is one blocking request: no job id, nothing to poll or cancel.
import { create } from "zustand";
import { factoryJson } from "@/lib/api";
import type { CleanBody } from "./cleanup";

/** What /clean answers. The panel uses only `file` (legacy); the face counts are never shown. */
export interface CleanResponse {
  ok?: boolean;
  stem?: string;
  file?: string;
  faces_in?: number;
  faces_out?: number;
}

/** POST /clean with the legacy body. A refusal or a script failure throws its message (factoryJson). */
export function postClean(body: CleanBody): Promise<CleanResponse> {
  return factoryJson<CleanResponse>("/clean", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/**
 * One clean at a time across Floaters and Carve. Legacy only held the Floaters button, so quick carve clicks sent
 * overlapping requests that each cut the same source file.
 */
export const useCleaning = create<{ busy: boolean }>(() => ({ busy: false }));
