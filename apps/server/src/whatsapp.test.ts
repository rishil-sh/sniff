import { expect, test } from "bun:test";
import { createHmac } from "node:crypto";

import { extractMessages, isValidSignature } from "./whatsapp";

const SECRET = "test-app-secret";
const sign = (body: string) =>
  `sha256=${createHmac("sha256", SECRET).update(body).digest("hex")}`;

const body = JSON.stringify({
  entry: [
    {
      changes: [
        {
          value: {
            messages: [
              {
                from: "15550001",
                id: "wamid.text",
                text: { body: "hello" },
                type: "text",
              },
              {
                from: "15550001",
                id: "wamid.img",
                image: { id: "media-1", mime_type: "image/jpeg" },
                type: "image",
              },
              {
                from: "15550001",
                id: "wamid.loc",
                location: { latitude: 1, longitude: 2 },
                type: "location",
              },
            ],
          },
        },
        { value: { statuses: [{ id: "wamid.out", status: "delivered" }] } },
      ],
    },
  ],
  object: "whatsapp_business_account",
});

test("accepts only Meta's signature for the exact body", () => {
  expect(isValidSignature(body, sign(body), SECRET)).toBe(true);
  expect(isValidSignature(`${body} `, sign(body), SECRET)).toBe(false);
  expect(isValidSignature(body, sign(body), "other-secret")).toBe(false);
  expect(isValidSignature(body, null, SECRET)).toBe(false);
  expect(isValidSignature(body, "sha256=short", SECRET)).toBe(false);
});

test("extracts text, media and unsupported messages, ignoring status updates", () => {
  const messages = extractMessages(body);
  expect(messages.map((m) => m.type)).toEqual(["text", "image", "location"]);
  expect(messages[0]?.text?.body).toBe("hello");
  expect(messages[1]?.image?.id).toBe("media-1");
  expect(extractMessages(JSON.stringify({ object: "page" }))).toEqual([]);
});
