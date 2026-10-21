// Asset names shared by Home (auto name, rename in Options) and the viewer (rename in the HUD).

/** The legacy slugStem (ix:4404-4419): lowercase words, no leading a/an/the, "-" joined, cut at 40 on a word. */
export function slugStem(name: string): string {
  const words =
    String(name || "")
      .replace(/\.[^.]+$/, "")
      .toLowerCase()
      .match(/[a-z0-9]+/g) || [];
  while (words.length && (words[0] === "a" || words[0] === "an" || words[0] === "the")) words.shift();
  let slug = words.join("-").replace(/^-+|-+$/g, "");
  if (slug.length > 40) {
    slug = slug.slice(0, 40);
    const dash = slug.lastIndexOf("-");
    if (dash > 0) slug = slug.slice(0, dash);
  }
  return slug;
}
