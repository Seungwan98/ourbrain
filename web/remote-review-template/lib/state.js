const LABELS = new Set(["negative", "crack", "uncertain", ""]);
const FORMULA_PREFIXES = new Set(["=", "+", "-", "@"]);
const NOTE_MAX_CHARACTERS = 80;
const NOTE_MAX_BYTES = 240;
const REVIEWER_MAX_CHARACTERS = 32;
const REVIEWER_MAX_BYTES = 96;

export function emptyState(datasetId) {
  return {
    schemaVersion: 1,
    datasetId,
    revision: 0,
    updatedAt: null,
    decisions: {},
    candidateVersions: {},
    conflicts: {},
  };
}

export function validateStoredState(value, datasetId, candidateIds) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("stored review state is not an object");
  }
  if (value.schemaVersion !== 1 || value.datasetId !== datasetId) {
    throw new Error("stored review state belongs to a different dataset");
  }
  if (!Number.isInteger(value.revision) || value.revision < 0) {
    throw new Error("stored review state has an invalid revision");
  }
  if (!value.decisions || typeof value.decisions !== "object") {
    throw new Error("stored review state has no decisions object");
  }
  if (!value.candidateVersions || typeof value.candidateVersions !== "object") {
    throw new Error("stored review state has no candidate versions object");
  }
  if (!value.conflicts || typeof value.conflicts !== "object") {
    throw new Error("stored review state has no conflicts object");
  }
  for (const [candidateId, decision] of Object.entries(value.decisions)) {
    if (!candidateIds.has(candidateId)) {
      throw new Error(`stored review state has an unknown candidate: ${candidateId}`);
    }
    validateDecision(decision);
  }
  return value;
}

function cleanText(value, maximumCharacters, maximumBytes, field) {
  if (value == null) {
    return "";
  }
  if (typeof value !== "string") {
    throw new TypeError(`${field} must be a string`);
  }
  const cleaned = value.trim();
  if (cleaned.length > maximumCharacters) {
    throw new RangeError(
      `${field} must be at most ${maximumCharacters} characters`,
    );
  }
  if (Buffer.byteLength(cleaned, "utf8") > maximumBytes) {
    throw new RangeError(`${field} contains too many UTF-8 bytes`);
  }
  return cleaned;
}

export function validateUpdate(value, candidateIds) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("request body must be a JSON object");
  }
  if (!candidateIds.has(value.candidateId)) {
    throw new RangeError("candidateId is unknown");
  }
  if (!LABELS.has(value.label)) {
    throw new RangeError("label must be negative, crack, uncertain, or blank");
  }
  if (!Number.isInteger(value.expectedRevision) || value.expectedRevision < 0) {
    throw new RangeError("expectedRevision must be a non-negative integer");
  }
  return {
    candidateId: value.candidateId,
    label: value.label,
    note: cleanText(value.note, NOTE_MAX_CHARACTERS, NOTE_MAX_BYTES, "note"),
    reviewer: cleanText(
      value.reviewer,
      REVIEWER_MAX_CHARACTERS,
      REVIEWER_MAX_BYTES,
      "reviewer",
    ),
    expectedRevision: value.expectedRevision,
  };
}

function validateDecision(value) {
  if (!value || typeof value !== "object" || !LABELS.has(value.label)) {
    throw new Error("stored decision is invalid");
  }
  cleanText(value.note, NOTE_MAX_CHARACTERS, NOTE_MAX_BYTES, "stored note");
  cleanText(
    value.reviewer,
    REVIEWER_MAX_CHARACTERS,
    REVIEWER_MAX_BYTES,
    "stored reviewer",
  );
  if (typeof value.updatedAt !== "string") {
    throw new Error("stored decision has no updatedAt timestamp");
  }
}

export function applyUpdate(state, update, now = new Date(), eventId = null) {
  if (update.expectedRevision !== state.revision) {
    const error = new Error("review state changed; reload before saving");
    error.code = "REVISION_CONFLICT";
    throw error;
  }
  const decisions = { ...state.decisions };
  if (update.label === "" && update.note === "") {
    delete decisions[update.candidateId];
  } else {
    decisions[update.candidateId] = {
      label: update.label,
      note: update.note,
      reviewer: update.reviewer,
      updatedAt: now.toISOString(),
    };
  }
  const candidateVersions = { ...state.candidateVersions };
  if (eventId) {
    candidateVersions[update.candidateId] = eventId;
  }
  const conflicts = { ...state.conflicts };
  delete conflicts[update.candidateId];
  return {
    ...state,
    revision: state.revision + 1,
    updatedAt: now.toISOString(),
    decisions,
    candidateVersions,
    conflicts,
  };
}

export function applyEvent(state, event, candidateIds) {
  if (typeof event.eventId !== "string" || !event.eventId) {
    throw new Error("stored review event has no eventId");
  }
  if (
    event.previousEventId !== null &&
    (typeof event.previousEventId !== "string" || !event.previousEventId)
  ) {
    throw new Error("stored review event has an invalid previousEventId");
  }
  const update = validateUpdate(event, candidateIds);
  if (update.expectedRevision > state.revision) {
    throw new Error("stored review event references a future revision");
  }
  const updatedAt = new Date(event.updatedAt);
  if (Number.isNaN(updatedAt.getTime())) {
    throw new Error("stored review event has an invalid timestamp");
  }

  const currentEventId = state.candidateVersions[update.candidateId] ?? null;
  const hasConflict = event.previousEventId !== currentEventId;
  const next = applyUpdate(
    state,
    { ...update, expectedRevision: state.revision },
    updatedAt,
    event.eventId,
  );
  if (hasConflict) {
    next.conflicts = {
      ...next.conflicts,
      [update.candidateId]: {
        candidateId: update.candidateId,
        expectedEventId: event.previousEventId,
        actualEventId: currentEventId,
        detectedAtRevision: next.revision,
      },
    };
  }
  return next;
}

export function replayEvents(datasetId, events, candidateIds) {
  return events.reduce(
    (state, event) => applyEvent(state, event, candidateIds),
    emptyState(datasetId),
  );
}

export function summarize(rows, decisions, conflicts = {}) {
  const summary = {
    total: rows.length,
    reviewed: 0,
    unreviewed: 0,
    negative: 0,
    crack: 0,
    uncertain: 0,
    conflicts: Object.keys(conflicts).length,
    conflictCandidateIds: Object.keys(conflicts).sort(),
    complete: false,
    bySplit: {},
  };
  for (const split of ["train", "val", "test"]) {
    summary.bySplit[split] = {
      total: 0,
      reviewed: 0,
      negative: 0,
      crack: 0,
      uncertain: 0,
      unreviewed: 0,
    };
  }
  for (const row of rows) {
    const split = summary.bySplit[row.target_split] ?? {
      total: 0,
      reviewed: 0,
      negative: 0,
      crack: 0,
      uncertain: 0,
      unreviewed: 0,
    };
    summary.bySplit[row.target_split] = split;
    split.total += 1;
    const label = decisions[row.id]?.label ?? "";
    if (label) {
      summary.reviewed += 1;
      summary[label] += 1;
      split.reviewed += 1;
      split[label] += 1;
    } else {
      summary.unreviewed += 1;
      split.unreviewed += 1;
    }
  }
  summary.complete =
    summary.reviewed === summary.total &&
    summary.total > 0 &&
    summary.conflicts === 0;
  return summary;
}

function protectSpreadsheetFormula(value) {
  const text = String(value ?? "");
  return FORMULA_PREFIXES.has(text[0]) ? `'${text}` : text;
}

function csvCell(value) {
  const protectedValue = protectSpreadsheetFormula(value);
  return `"${protectedValue.replaceAll('"', '""')}"`;
}

export function buildCsv(rows, fields, decisions) {
  const lines = [fields.map(csvCell).join(",")];
  for (const row of rows) {
    const decision = decisions[row.id] ?? {};
    const exported = {
      ...row,
      review_label: decision.label ?? "",
      review_note: decision.note ?? "",
      reviewer: decision.reviewer ?? "",
      reviewed_at: decision.updatedAt ?? "",
    };
    lines.push(fields.map((field) => csvCell(exported[field])).join(","));
  }
  return `\uFEFF${lines.join("\r\n")}\r\n`;
}

export function publicState(state, rows) {
  return {
    datasetId: state.datasetId,
    revision: state.revision,
    updatedAt: state.updatedAt,
    decisions: state.decisions,
    conflicts: state.conflicts,
    summary: summarize(rows, state.decisions, state.conflicts),
  };
}
