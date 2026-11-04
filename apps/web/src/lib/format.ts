const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

export const formatDateTime = (value: Date | string) =>
  dateTimeFormat.format(new Date(value));

const STARTS_WITH_LETTER = /^\p{L}/u;

export const initials = (name: string) =>
  name
    .split(" ")
    .filter((part) => STARTS_WITH_LETTER.test(part))
    .map((part) => part.charAt(0))
    .join("")
    .slice(0, 2)
    .toUpperCase();
