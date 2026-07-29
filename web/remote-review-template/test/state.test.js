import assert from "node:assert/strict";
import test from "node:test";

import {
  applyUpdate,
  buildCsv,
  emptyState,
  replayEvents,
  summarize,
  validateStoredState,
  validateUpdate,
} from "../lib/state.js";

const rows = [
  { id: "a", candidate_path: "a.png", target_split: "train" },
  { id: "b", candidate_path: "b.png", target_split: "val" },
  { id: "c", candidate_path: "c.png", target_split: "test" },
];
const ids = new Set(rows.map((row) => row.id));

test("updates are revision checked and summarized by split", () => {
  const update = validateUpdate(
    {
      candidateId: "a",
      label: "negative",
      note: "clear",
      reviewer: "domain expert",
      expectedRevision: 0,
    },
    ids,
  );
  const state = applyUpdate(
    emptyState("dataset"),
    update,
    new Date("2026-07-29T00:00:00Z"),
  );
  const summary = summarize(rows, state.decisions);

  assert.equal(state.revision, 1);
  assert.equal(summary.reviewed, 1);
  assert.equal(summary.bySplit.train.negative, 1);
  assert.equal(summary.complete, false);
  assert.throws(
    () => applyUpdate(state, { ...update, expectedRevision: 0 }),
    /changed/,
  );
});

test("stored state rejects unknown candidates", () => {
  const state = {
    ...emptyState("dataset"),
    decisions: {
      unknown: {
        label: "negative",
        note: "",
        reviewer: "",
        updatedAt: "2026-07-29T00:00:00Z",
      },
    },
  };
  assert.throws(
    () => validateStoredState(state, "dataset", ids),
    /unknown candidate/,
  );
});

test("review text is bounded before it can exceed the event pathname limit", () => {
  assert.throws(
    () =>
      validateUpdate(
        {
          candidateId: "a",
          label: "uncertain",
          note: "x".repeat(81),
          reviewer: "",
          expectedRevision: 0,
        },
        ids,
      ),
    /at most 80 characters/,
  );
  assert.throws(
    () =>
      validateUpdate(
        {
          candidateId: "a",
          label: "uncertain",
          note: "",
          reviewer: "x".repeat(33),
          expectedRevision: 0,
        },
        ids,
      ),
    /at most 32 characters/,
  );
  assert.doesNotThrow(() =>
    validateUpdate(
      {
        candidateId: "a",
        label: "uncertain",
        note: "한".repeat(80),
        reviewer: "한".repeat(32),
        expectedRevision: 0,
      },
      ids,
    ),
  );
});

test("CSV export is strict-import compatible and formula safe", () => {
  const csv = buildCsv(
    [{ ...rows[0], source_image_path: "=unsafe" }],
    ["candidate_path", "source_image_path", "review_label", "review_note"],
    {
      a: {
        label: "negative",
        note: "checked",
        reviewer: "reviewer",
        updatedAt: "2026-07-29T00:00:00Z",
      },
    },
  );

  assert.match(csv, /^\uFEFF/);
  assert.match(csv, /"negative"/);
  assert.match(csv, /"'=unsafe"/);
});

function event(eventId, candidateId, expectedRevision, previousEventId, label) {
  return {
    eventId,
    candidateId,
    expectedRevision,
    previousEventId,
    label,
    note: "",
    reviewer: "reviewer",
    updatedAt: `2026-07-29T00:00:0${expectedRevision}Z`,
  };
}

test("concurrent decisions for different candidates merge without conflict", () => {
  const state = replayEvents(
    "dataset",
    [
      event("event-a", "a", 0, null, "negative"),
      event("event-b", "b", 0, null, "crack"),
    ],
    ids,
  );

  assert.equal(state.revision, 2);
  assert.equal(state.decisions.a.label, "negative");
  assert.equal(state.decisions.b.label, "crack");
  assert.deepEqual(state.conflicts, {});
});

test("same-candidate concurrent decisions require an explicit resolution", () => {
  const conflicted = replayEvents(
    "dataset",
    [
      event("event-a", "a", 0, null, "negative"),
      event("event-b", "a", 0, null, "crack"),
    ],
    ids,
  );

  assert.equal(conflicted.decisions.a.label, "crack");
  assert.equal(conflicted.conflicts.a.candidateId, "a");
  assert.equal(summarize(rows, conflicted.decisions, conflicted.conflicts).complete, false);

  const resolved = replayEvents(
    "dataset",
    [
      event("event-a", "a", 0, null, "negative"),
      event("event-b", "a", 0, null, "crack"),
      event("event-c", "a", 2, "event-b", "uncertain"),
    ],
    ids,
  );
  assert.equal(resolved.decisions.a.label, "uncertain");
  assert.deepEqual(resolved.conflicts, {});
});
