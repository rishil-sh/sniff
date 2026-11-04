import Link from "next/link";

import { Logo } from "@/components/shell/logo";

const AuthLayout = ({ children }: { children: React.ReactNode }) => (
  <main className="grid min-h-svh lg:grid-cols-2">
    <div className="flex flex-col gap-6 p-6 md:p-10">
      <Link
        className="font-heading flex items-center gap-2 font-semibold"
        href="/"
      >
        <Logo />
        Sniff
      </Link>
      <div className="flex flex-1 items-center justify-center">
        <div className="w-full max-w-sm">{children}</div>
      </div>
    </div>
    <div className="bg-wa-wallpaper hidden flex-col justify-end gap-3 border-l p-10 lg:flex">
      <p className="font-heading text-3xl font-semibold text-balance">
        Is this AI?
      </p>
      <p className="text-muted-foreground max-w-md text-balance">
        Send a photo, a voice note or a paragraph of text, here or on WhatsApp,
        and Sniff tells you whether AI made it, with the evidence behind the
        answer.
      </p>
    </div>
  </main>
);

export default AuthLayout;
