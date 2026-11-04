import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    background_color: "#fdf7f2",
    description: "Forward anything. Sniff checks if it's real.",
    display: "standalone",
    icons: [
      { sizes: "192x192", src: "/brand/icon-192.png", type: "image/png" },
      { sizes: "512x512", src: "/brand/icon-512.png", type: "image/png" },
    ],
    name: "Sniff",
    short_name: "Sniff",
    start_url: "/",
    theme_color: "#1a1718",
  };
}
