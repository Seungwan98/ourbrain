import { list, put } from "@vercel/blob";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { CANDIDATES, DATASET_ID } from "../lib/dataset.js";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const prefix = `review-images/${DATASET_ID}/`;

async function existingPathnames() {
  const pathnames = new Set();
  let cursor;
  do {
    const page = await list({ prefix, limit: 1000, cursor });
    for (const blob of page.blobs) {
      pathnames.add(blob.pathname);
    }
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return pathnames;
}

async function main() {
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    throw new Error("BLOB_READ_WRITE_TOKEN is required");
  }
  const existing = await existingPathnames();
  const pending = CANDIDATES.filter((row) => !existing.has(row.imageBlobPath));
  let uploaded = 0;
  const queue = [...pending];
  const workers = Array.from({ length: Math.min(6, queue.length) }, async () => {
    while (queue.length) {
      const candidate = queue.shift();
      const filename = candidate.imageBlobPath.slice(prefix.length);
      const body = await readFile(join(root, "private-candidates", filename));
      await put(candidate.imageBlobPath, body, {
        access: "private",
        addRandomSuffix: false,
        // Pathnames are content-addressed by the image SHA-256. Re-uploading
        // identical bytes is safe and avoids failures from a briefly stale list().
        allowOverwrite: true,
        contentType: candidate.imageContentType,
      });
      uploaded += 1;
      if (uploaded % 20 === 0 || uploaded === pending.length) {
        process.stdout.write(`uploaded ${uploaded}/${pending.length}\n`);
      }
    }
  });
  await Promise.all(workers);
  process.stdout.write(
    JSON.stringify(
      {
        datasetId: DATASET_ID,
        candidates: CANDIDATES.length,
        uploaded,
        existing: existing.size,
      },
      null,
      2,
    ) + "\n",
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
