"use client";

import { Toaster } from "@sniff/ui/components/sonner";
import { TooltipProvider } from "@sniff/ui/components/tooltip";
import { QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

import { getQueryClient } from "@/utils/orpc";

import { ThemeProvider } from "./theme-provider";

const Providers = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider
    attribute="class"
    defaultTheme="dark"
    enableSystem
    disableTransitionOnChange
  >
    <QueryClientProvider client={getQueryClient()}>
      <TooltipProvider>{children}</TooltipProvider>
      <ReactQueryDevtools />
    </QueryClientProvider>
    <Toaster richColors />
  </ThemeProvider>
);

export default Providers;
