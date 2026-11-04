"use client";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@sniff/ui/components/sidebar";
import { HistoryIcon, ScanSearchIcon } from "lucide-react";
import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Logo } from "./logo";
import { NavUser } from "./nav-user";

const NAV = [
  { href: "/", icon: ScanSearchIcon, title: "Check" },
  { href: "/history", icon: HistoryIcon, title: "History" },
] as const satisfies readonly {
  href: Route;
  icon: typeof ScanSearchIcon;
  title: string;
}[];

// On phones the sidebar is a sheet over the page, so following a link closes it.
const NavLink = ({
  active,
  item,
}: {
  active: boolean;
  item: (typeof NAV)[number];
}) => {
  const { setOpenMobile } = useSidebar();

  return (
    <SidebarMenuButton
      isActive={active}
      onClick={() => setOpenMobile(false)}
      render={<Link href={item.href} />}
      tooltip={item.title}
    >
      <item.icon />
      <span>{item.title}</span>
    </SidebarMenuButton>
  );
};

// Trimmed from logicsense's shell: full-width top bar, icon-collapsible sidebar below it.
export const AppShell = ({
  children,
  defaultOpen,
}: {
  children: React.ReactNode;
  defaultOpen: boolean;
}) => {
  const pathname = usePathname();
  const active = NAV.find((item) => item.href === pathname);

  return (
    <SidebarProvider
      className="flex-col [--header-height:3rem]"
      defaultOpen={defaultOpen}
      style={
        // SAFETY: CSS custom properties aren't part of React.CSSProperties.
        { "--sidebar-width": "13rem" } as React.CSSProperties
      }
    >
      <header className="bg-sidebar sticky top-0 z-20 flex h-(--header-height) w-full shrink-0 items-center gap-2 border-b px-3">
        <SidebarTrigger className="md:hidden" />
        <Link
          className="font-heading flex items-center gap-2 px-1 font-semibold"
          href="/"
        >
          <Logo />
          Sniff
        </Link>
        {active && (
          <>
            <span aria-hidden className="text-muted-foreground/40 select-none">
              /
            </span>
            <span className="min-w-0 truncate text-sm">{active.title}</span>
          </>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          <NavUser />
        </div>
      </header>
      <div className="flex flex-1">
        <Sidebar
          className="top-(--header-height) h-[calc(100svh-var(--header-height))]!"
          collapsible="icon"
        >
          <SidebarContent>
            <SidebarGroup>
              <SidebarMenu>
                {NAV.map((item) => (
                  <SidebarMenuItem key={item.href}>
                    <NavLink active={active?.href === item.href} item={item} />
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroup>
          </SidebarContent>
          <SidebarFooter>
            <SidebarTrigger />
          </SidebarFooter>
        </Sidebar>
        <SidebarInset className="min-w-0">
          <main className="flex flex-1 flex-col gap-6 p-4 has-data-[slot=product-layout]:p-0 md:p-6">
            {children}
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
};
