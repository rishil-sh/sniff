import { cookies } from "next/headers";

// Written by the shadcn SidebarProvider; read on the server so the first paint matches.
export const sidebarDefaultOpen = async () => {
  const cookieStore = await cookies();

  return cookieStore.get("sidebar_state")?.value !== "false";
};
