"use client";

import { useSyncExternalStore } from "react";

import { authClient } from "./auth-client";

// oxlint-disable-next-line eslint/no-empty-function -- nothing to subscribe to; only the server and client snapshots differ
const noopSubscribe = () => () => {};

// useSession, but pending until hydration: the client can already hold a
// cached session the server render never had, which breaks hydration.
export const useSession = () => {
  const result = authClient.useSession();
  const hydrated = useSyncExternalStore(
    noopSubscribe,
    () => true,
    () => false
  );

  return hydrated ? result : { ...result, data: null, isPending: true };
};
