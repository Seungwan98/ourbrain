import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  bundledCandidateFilename,
  readBundledCandidate,
} from "../lib/candidate-file.js";

test("reads a content-addressed candidate from the bundled fallback", async () => {
  const root = await mkdtemp(join(tmpdir(), "ourbrain-candidate-"));
  const candidate = {
    imageBlobPath: "review-images/dataset/candidate-deadbeef.png",
  };
  try {
    await writeFile(join(root, "candidate-deadbeef.png"), "image-bytes");
    const body = await readBundledCandidate(candidate, { root });

    assert.equal(bundledCandidateFilename(candidate), "candidate-deadbeef.png");
    assert.equal(body.toString(), "image-bytes");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("returns null when the bundled candidate is absent", async () => {
  const root = await mkdtemp(join(tmpdir(), "ourbrain-candidate-"));
  try {
    const body = await readBundledCandidate(
      { imageBlobPath: "review-images/dataset/missing.png" },
      { root },
    );
    assert.equal(body, null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects unsupported candidate paths", () => {
  assert.throws(
    () => bundledCandidateFilename({ imageBlobPath: "review-images/data.txt" }),
    /invalid/,
  );
});
