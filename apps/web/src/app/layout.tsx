import type { Metadata, Viewport } from "next";
import { Inter, Manrope, Source_Code_Pro } from "next/font/google";

import "../index.css";
import Providers from "@/components/providers";
import PwaRegistration from "@/components/pwa-registration";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope" });
const sourceCodePro = Source_Code_Pro({
  subsets: ["latin"],
  variable: "--font-source-code-pro",
});

export const metadata: Metadata = {
  appleWebApp: { capable: true, statusBarStyle: "default", title: "Sniff" },
  description:
    "Is this AI? Check images, voice notes and text on the web or on WhatsApp.",
  title: { default: "Sniff", template: "%s · Sniff" },
};

export const viewport: Viewport = {
  themeColor: [
    { color: "#ffffff", media: "(prefers-color-scheme: light)" },
    { color: "#1c1c1c", media: "(prefers-color-scheme: dark)" },
  ],
};

const RootLayout = ({ children }: Readonly<{ children: React.ReactNode }>) => (
  <html lang="en" suppressHydrationWarning>
    <body
      className={`${inter.variable} ${manrope.variable} ${sourceCodePro.variable} antialiased`}
    >
      <PwaRegistration />
      <Providers>{children}</Providers>
    </body>
  </html>
);

export default RootLayout;
