import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    background_color: "#1c1c1c",
    description: "Sniff — is this AI?",
    display: "standalone",
    icons: [
      {
        sizes: "192x192",
        src: "/favicon/web-app-manifest-192x192.png",
        type: "image/png",
      },
      {
        sizes: "512x512",
        src: "/favicon/web-app-manifest-512x512.png",
        type: "image/png",
      },
    ],
    name: "Sniff",
    short_name: "Sniff",
    start_url: "/",
    theme_color: "#1c1c1c",
  };
}
