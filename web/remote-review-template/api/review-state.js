import { list, put } from "@vercel/blob";
import { randomUUID } from "node:crypto";

import { authorized, jsonResponse } from "../lib/auth.js";
import { CANDIDATES, DATASET_ID, EXPORT_FIELDS } from "../lib/dataset.js";
import {
  applyUpdate,
  buildCsv,
  emptyState,
  applyEvent,
  publicState,
  validateUpdate,
} from "../lib/state.js";

const EVENT_PREFIX = `review-events-v1/${DATASET_ID}/`;
const CANDIDATE_IDS = new Set(CANDIDATES.map((row) => row.id));
const MAX_EVENT_PATHNAME_LENGTH = 900;

function response(body, status = 200, headers = {}) {
  return new Response(body, {
    status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8",
      ...headers,
    },
  });
}

function json(value, status = 200) {
  return jsonResponse(value, status);
}

async function readState() {
  const blobs = [];
  let cursor;
  do {
    const page = await list({
      prefix: EVENT_PREFIX,
      limit: 1000,
      cursor,
    });
    blobs.push(...page.blobs);
    cursor = page.hasMore ? page.cursor : undefined;
    if (blobs.length > 5000) {
      throw new Error("remote review event limit exceeded");
    }
  } while (cursor);

  let state = emptyState(DATASET_ID);
  const ordered = blobs.sort((left, right) => {
    const uploaded = new Date(left.uploadedAt) - new Date(right.uploadedAt);
    return uploaded || left.pathname.localeCompare(right.pathname);
  });
  for (const blob of ordered) {
    const filename = blob.pathname.slice(EVENT_PREFIX.length).replace(/\.json$/, "");
    const parts = filename.split(".");
    if (parts.length !== 3) {
      throw new Error(`invalid remote review event pathname: ${blob.pathname}`);
    }
    let event;
    try {
      event = JSON.parse(Buffer.from(parts[2], "base64url").toString("utf-8"));
    } catch {
      throw new Error(`invalid remote review event payload: ${blob.pathname}`);
    }
    event.eventId = parts[1];
    try {
      state = applyEvent(state, event, CANDIDATE_IDS);
    } catch (error) {
      throw new Error(
        `invalid remote review event ${blob.pathname}: ${error.message}`,
      );
    }
  }
  return state;
}

async function appendEvent(update, updatedAt, eventId, previousEventId) {
  const event = {
    candidateId: update.candidateId,
    label: update.label,
    note: update.note,
    reviewer: update.reviewer,
    expectedRevision: update.expectedRevision,
    previousEventId,
    updatedAt: updatedAt.toISOString(),
  };
  const encoded = Buffer.from(JSON.stringify(event), "utf-8").toString("base64url");
  const timestamp = String(updatedAt.getTime()).padStart(13, "0");
  const pathname = `${EVENT_PREFIX}${timestamp}.${eventId}.${encoded}.json`;
  if (pathname.length > MAX_EVENT_PATHNAME_LENGTH) {
    throw new Error("remote review event pathname exceeds the safe limit");
  }
  return put(pathname, "event", {
    access: "private",
    addRandomSuffix: false,
    contentType: "application/json",
  });
}

async function updateState(update) {
  const state = await readState();
  if (update.expectedRevision !== state.revision) {
    return { conflict: true, state };
  }
  const updatedAt = new Date();
  const eventId = randomUUID();
  const previousEventId = state.candidateVersions[update.candidateId] ?? null;
  const next = applyUpdate(state, update, updatedAt, eventId);
  await appendEvent(update, updatedAt, eventId, previousEventId);
  return { conflict: false, state: next };
}

async function handle(request) {
  if (!process.env.BLOB_READ_WRITE_TOKEN || !process.env.REVIEW_TOKEN) {
    return json({ error: "review service is not configured" }, 503);
  }
  if (!authorized(request)) {
    return json({ error: "invalid review access code" }, 401);
  }

  const method = request.method.toUpperCase();
  const url = new URL(request.url);
  if (method === "GET") {
    const state = await readState();
    if (url.searchParams.get("format") === "csv") {
      const current = publicState(state, CANDIDATES);
      if (!current.summary.complete) {
        return json(
          {
            error: "all candidates must be reviewed before CSV export",
            ...current,
          },
          409,
        );
      }
      const csv = buildCsv(CANDIDATES, EXPORT_FIELDS, state.decisions);
      return response(csv, 200, {
        "Content-Disposition":
          'attachment; filename="negative_review_reviewed.csv"',
        "Content-Type": "text/csv; charset=utf-8",
      });
    }
    return json({
      ...publicState(state, CANDIDATES),
      candidates: CANDIDATES.map(({ imageBlobPath, ...candidate }) => candidate),
    });
  }

  if (method === "POST") {
    const contentLength = Number(request.headers.get("content-length") ?? 0);
    if (contentLength > 4096) {
      return json({ error: "request body is too large" }, 413);
    }
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "request body must be valid JSON" }, 400);
    }
    let update;
    try {
      update = validateUpdate(body, CANDIDATE_IDS);
    } catch (error) {
      return json({ error: error.message }, 400);
    }
    const result = await updateState(update);
    if (result.conflict) {
      return json(
        {
          error: "review state changed; latest state returned",
          ...publicState(result.state, CANDIDATES),
        },
        409,
      );
    }
    return json(publicState(result.state, CANDIDATES));
  }

  return json({ error: "method not allowed" }, 405);
}

export default {
  fetch(request) {
    return handle(request).catch((error) => {
      console.error("remote review failure", error);
      return json({ error: "review service failed" }, 500);
    });
  },
};
