import { headers } from "next/headers";
import { redirect } from "next/navigation";

import { AppShell } from "@/components/shell/app-shell";
import { authClient } from "@/lib/auth-client";
import { sidebarDefaultOpen } from "@/lib/sidebar-state";

const AppLayout = async ({ children }: { children: React.ReactNode }) => {
  const session = await authClient.getSession({
    fetchOptions: { headers: await headers(), throw: true },
  });
  if (!session?.user) {
    redirect("/login");
  }
  return (
    <AppShell defaultOpen={await sidebarDefaultOpen()}>{children}</AppShell>
  );
};

export default AppLayout;
