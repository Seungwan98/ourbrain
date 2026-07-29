import { timingSafeEqual } from "node:crypto";

export function authorized(request) {
  const expected = process.env.REVIEW_TOKEN;
  const provided =
    request.headers.get("authorization")?.replace(/^Bearer\s+/i, "") ?? "";
  if (!expected || !provided) {
    return false;
  }
  const expectedBytes = Buffer.from(expected);
  const providedBytes = Buffer.from(provided);
  return (
    expectedBytes.length === providedBytes.length &&
    timingSafeEqual(expectedBytes, providedBytes)
  );
}

export function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8",
    },
  });
}
