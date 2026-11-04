import { Fragment } from "react";

export interface WhatsAppToken {
  kind: "bold" | "code" | "italic" | "link" | "pre" | "strike" | "text";
  text: string;
}

const MARKUP =
  /```[\s\S]+?```|`[^`\n]+`|\*[^*\n]+\*|_[^_\n]+_|~[^~\n]+~|https?:\/\/[^\s<>]+[^\s<>.,;:!?)\]'"]/gu;

const DELIMITERS = [
  { kind: "pre", marker: "```" },
  { kind: "code", marker: "`" },
  { kind: "bold", marker: "*" },
  { kind: "italic", marker: "_" },
  { kind: "strike", marker: "~" },
] as const;

const tokenFor = (raw: string): WhatsAppToken => {
  const delimiter = DELIMITERS.find(({ marker }) => raw.startsWith(marker));

  return delimiter
    ? {
        kind: delimiter.kind,
        text: raw.slice(delimiter.marker.length, -delimiter.marker.length),
      }
    : { kind: "link", text: raw };
};

// WhatsApp's own markup (not Markdown). Styles don't nest; neither do they in most clients.
export const parseWhatsApp = (text: string): WhatsAppToken[] => {
  const tokens: WhatsAppToken[] = [];
  let last = 0;

  for (const match of text.matchAll(MARKUP)) {
    if (match.index > last) {
      tokens.push({ kind: "text", text: text.slice(last, match.index) });
    }

    tokens.push(tokenFor(match[0]));
    last = match.index + match[0].length;
  }

  if (last < text.length) {
    tokens.push({ kind: "text", text: text.slice(last) });
  }

  return tokens;
};

const Token = ({ token }: { token: WhatsAppToken }) => {
  switch (token.kind) {
    case "bold": {
      return <strong>{token.text}</strong>;
    }
    case "italic": {
      return <em>{token.text}</em>;
    }
    case "strike": {
      return <s>{token.text}</s>;
    }
    case "code": {
      return <code className="font-mono">{token.text}</code>;
    }
    case "pre": {
      return <pre className="font-mono whitespace-pre-wrap">{token.text}</pre>;
    }
    case "link": {
      return (
        <a
          className="underline underline-offset-2"
          href={token.text}
          rel="noopener noreferrer"
          target="_blank"
        >
          {token.text}
        </a>
      );
    }
    default: {
      return token.text;
    }
  }
};

export const WhatsAppText = ({ text }: { text: string }) => (
  <span className="whitespace-pre-wrap">
    {parseWhatsApp(text).map((token, index) => (
      // Tokens have no identity beyond their position in this string.
      // oxlint-disable-next-line react/no-array-index-key
      <Fragment key={index}>
        <Token token={token} />
      </Fragment>
    ))}
  </span>
);
