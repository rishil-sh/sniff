"use client";

import { useEffect } from "react";

const register = async () => {
  try {
    await navigator.serviceWorker.register("/sw.js", {
      scope: "/",
      updateViaCache: "none",
    });
  } catch (error) {
    console.error("Service worker registration failed:", error);
  }
};

const PwaRegistration = () => {
  useEffect(() => {
    if ("serviceWorker" in navigator) {
      register();
    }
  }, []);

  return null;
};

export default PwaRegistration;
