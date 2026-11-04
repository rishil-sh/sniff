import { cn } from "@sniff/ui/lib/utils";
import Image from "next/image";

/** Sherlock, Sniff's beagle, on his dark app-icon tile. */
export const Logo = ({
  className,
  size = 28,
}: {
  className?: string;
  size?: number;
}) => (
  <Image
    alt=""
    className={cn("shrink-0", className)}
    height={size}
    priority
    src="/brand/sherlock.png"
    width={size}
  />
);

/** Logo plus the wordmark, as on the brand board. */
export const Wordmark = ({ className }: { className?: string }) => (
  <span
    className={cn(
      "font-heading flex items-center gap-2 font-extrabold tracking-tight",
      className
    )}
  >
    <Logo />
    Sniff
  </span>
);
