import { mergeProps } from "@base-ui/react/merge-props";
import { useRender } from "@base-ui/react/use-render";
import { cn } from "@sniff/ui/lib/utils";
import { cva } from "class-variance-authority";
import type { VariantProps } from "class-variance-authority";

// Verdict variants are Sniff's verdict pills (docs/brand/brand-board.png), with Sherlock's face as the icon.
const badgeVariants = cva(
  "group/badge focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 inline-flex h-5 w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-full border border-transparent px-1.5 py-0.5 font-mono text-[11px] font-medium tracking-wider whitespace-nowrap uppercase transition-all focus-visible:ring-[3px] has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&>svg]:pointer-events-none [&>svg]:size-3!",
  {
    defaultVariants: {
      variant: "default",
    },
    variants: {
      variant: {
        "declared-ai":
          "bg-verdict-declared-soft text-verdict-declared font-sans text-xs font-semibold tracking-normal normal-case",
        default: "bg-primary text-primary-foreground [a]:hover:bg-primary/80",
        destructive:
          "bg-destructive/10 text-destructive focus-visible:ring-destructive/20 dark:bg-destructive/20 dark:focus-visible:ring-destructive/40 [a]:hover:bg-destructive/20",
        ghost:
          "hover:bg-muted hover:text-muted-foreground dark:hover:bg-muted/50",
        inconclusive:
          "bg-verdict-unsure-soft text-verdict-unsure font-sans text-xs font-semibold tracking-normal normal-case",
        "likely-ai":
          "bg-verdict-ai-soft text-verdict-ai font-sans text-xs font-semibold tracking-normal normal-case",
        link: "text-primary underline-offset-4 hover:underline",
        "no-ai-signal":
          "bg-verdict-clear-soft text-verdict-clear font-sans text-xs font-semibold tracking-normal normal-case",
        outline:
          "border-border text-foreground [a]:hover:bg-muted [a]:hover:text-muted-foreground",
        secondary:
          "bg-secondary text-secondary-foreground [a]:hover:bg-secondary/80",
      },
    },
  }
);

const Badge = ({
  className,
  variant = "default",
  render,
  ...props
}: useRender.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) =>
  useRender({
    defaultTagName: "span",
    props: mergeProps<"span">(
      {
        className: cn(badgeVariants({ variant }), className),
      },
      props
    ),
    render,
    state: {
      slot: "badge",
      variant,
    },
  });

export { Badge, badgeVariants };
