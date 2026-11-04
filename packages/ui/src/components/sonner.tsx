"use client";

import {
  CircleCheckIcon,
  InfoIcon,
  Loader2Icon,
  OctagonXIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useTheme } from "next-themes";
import { Toaster as Sonner } from "sonner";
import type { ToasterProps } from "sonner";

declare module "react" {
  interface CSSProperties {
    [key: `--${string}`]: string;
  }
}

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme } = useTheme();

  return (
    <Sonner
      theme={theme === "light" || theme === "dark" ? theme : "system"}
      className="group"
      icons={{
        error: <OctagonXIcon className="size-4" />,
        info: <InfoIcon className="size-4" />,
        loading: <Loader2Icon className="size-4 animate-spin" />,
        success: <CircleCheckIcon className="size-4" />,
        warning: <TriangleAlertIcon className="size-4" />,
      }}
      style={{
        "--border-radius": "var(--radius)",
        "--normal-bg": "var(--popover)",
        "--normal-border": "var(--border)",
        "--normal-text": "var(--popover-foreground)",
      }}
      {...props}
    />
  );
};

export { Toaster };
