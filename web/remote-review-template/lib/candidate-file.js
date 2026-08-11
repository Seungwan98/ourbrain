import { readFile } from "node:fs/promises";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_CANDIDATE_ROOT = join(
  dirname(dirname(fileURLToPath(import.meta.url))),
  "private-candidates",
);
const SUPPORTED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp"]);

export function bundledCandidateFilename(candidate) {
  const pathname = candidate?.imageBlobPath;
  if (typeof pathname !== "string" || !pathname) {
    throw new TypeError("candidate image pathname is missing");
  }
  const filename = pathname.split("/").at(-1) ?? "";
  if (!filename || !SUPPORTED_EXTENSIONS.has(extname(filename).toLowerCase())) {
    throw new TypeError("candidate image pathname is invalid");
  }
  return filename;
}

export async function readBundledCandidate(
  candidate,
  { root = DEFAULT_CANDIDATE_ROOT } = {},
) {
  const filename = bundledCandidateFilename(candidate);
  try {
    return await readFile(join(root, filename));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}
