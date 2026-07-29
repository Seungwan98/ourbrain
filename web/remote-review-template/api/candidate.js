import { get } from "@vercel/blob";

import { authorized, jsonResponse } from "../lib/auth.js";
import { CANDIDATES } from "../lib/dataset.js";

const CANDIDATE_BY_ID = new Map(CANDIDATES.map((row) => [row.id, row]));

async function handle(request) {
  if (!process.env.BLOB_READ_WRITE_TOKEN || !process.env.REVIEW_TOKEN) {
    return jsonResponse({ error: "review service is not configured" }, 503);
  }
  if (!authorized(request)) {
    return jsonResponse({ error: "invalid review access code" }, 401);
  }
  if (request.method.toUpperCase() !== "GET") {
    return jsonResponse({ error: "method not allowed" }, 405);
  }
  const candidateId = new URL(request.url).searchParams.get("id") ?? "";
  const candidate = CANDIDATE_BY_ID.get(candidateId);
  if (!candidate) {
    return jsonResponse({ error: "candidate not found" }, 404);
  }
  const blob = await get(candidate.imageBlobPath, { access: "private" });
  if (!blob || blob.statusCode !== 200 || !blob.stream) {
    return jsonResponse({ error: "candidate image is not available" }, 404);
  }
  return new Response(blob.stream, {
    status: 200,
    headers: {
      "Cache-Control": "private, no-store",
      "Content-Type": blob.blob.contentType ?? "application/octet-stream",
      "ETag": `"${candidate.imageSha256}"`,
      "X-Content-SHA256": candidate.imageSha256,
      "X-Content-Type-Options": "nosniff",
    },
  });
}

export default {
  fetch(request) {
    return handle(request).catch((error) => {
      console.error("candidate delivery failure", error);
      return jsonResponse({ error: "candidate delivery failed" }, 500);
    });
  },
};
