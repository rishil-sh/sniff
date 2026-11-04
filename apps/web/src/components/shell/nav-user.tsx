"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@sniff/ui/components/avatar";
import { Button } from "@sniff/ui/components/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@sniff/ui/components/dropdown-menu";
import { Skeleton } from "@sniff/ui/components/skeleton";
import { LogOutIcon, UserIcon } from "lucide-react";
import type { Route } from "next";
import { useTheme } from "next-themes";
import Link from "next/link";

import { authClient } from "@/lib/auth-client";
import { initials } from "@/lib/format";
import { useSession } from "@/lib/use-session";

export const NavUser = ({ profileHref }: { profileHref?: Route }) => {
  const { data: session, isPending } = useSession();
  const { setTheme, theme } = useTheme();

  if (isPending || !session) {
    return <Skeleton className="size-7" />;
  }

  const { user } = session;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={<Button aria-label="Account" size="icon-sm" variant="ghost" />}
      >
        <Avatar size="sm">
          <AvatarImage alt={user.name} src={user.image ?? undefined} />
          <AvatarFallback>{initials(user.name)}</AvatarFallback>
        </Avatar>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-56">
        <DropdownMenuGroup>
          <DropdownMenuLabel className="flex flex-col">
            <span className="text-foreground font-medium">{user.name}</span>
            <span>{user.email}</span>
          </DropdownMenuLabel>
          {profileHref && (
            <DropdownMenuItem render={<Link href={profileHref} />}>
              <UserIcon />
              Profile
            </DropdownMenuItem>
          )}
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <DropdownMenuGroup>
          <DropdownMenuLabel>Theme</DropdownMenuLabel>
          <DropdownMenuRadioGroup onValueChange={setTheme} value={theme}>
            <DropdownMenuRadioItem value="light">Light</DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="dark">Dark</DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="system">System</DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={async () => {
            await authClient.signOut();
            window.location.assign("/login");
          }}
          variant="destructive"
        >
          <LogOutIcon />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
