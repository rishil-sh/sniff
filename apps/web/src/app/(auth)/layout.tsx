import Image from "next/image";
import Link from "next/link";

import {
  MOOD_ORDER,
  MOODS,
  SherlockPose,
} from "@/components/check/verdict-badge";
import { Wordmark } from "@/components/shell/logo";

const AuthLayout = ({ children }: { children: React.ReactNode }) => (
  <main className="grid min-h-svh lg:grid-cols-2">
    <div className="flex flex-col gap-6 p-6 md:p-10">
      <Link aria-label="Sniff home" href="/">
        <Wordmark />
      </Link>
      <div className="flex flex-1 items-center justify-center">
        <div className="w-full max-w-sm">{children}</div>
      </div>
    </div>
    <aside className="bg-secondary hidden flex-col justify-center gap-10 border-l p-10 lg:flex xl:p-16">
      <div className="flex items-center gap-6">
        <Image
          alt="Sherlock, Sniff's beagle detective"
          className="size-28 shrink-0 xl:size-36"
          height={144}
          priority
          src="/brand/sherlock.png"
          width={144}
        />
        <div className="flex flex-col gap-2">
          <p className="font-heading text-4xl font-extrabold tracking-tight text-balance xl:text-5xl">
            Forward anything.
          </p>
          <p className="text-muted-foreground font-heading text-xl font-semibold text-balance xl:text-2xl">
            Sniff checks if it&apos;s real.
          </p>
        </div>
      </div>
      <ul className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        {MOOD_ORDER.map((mood) => (
          <li className="bg-card overflow-hidden rounded-2xl border" key={mood}>
            <div
              className={`flex justify-center pt-3 ${MOODS[mood].className}`}
            >
              <SherlockPose
                className="h-32 w-auto"
                height={300}
                pose={mood}
                width={290}
              />
            </div>
            <p className="p-2 text-center">
              <span
                className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap ${MOODS[mood].className}`}
              >
                {MOODS[mood].label}
              </span>
            </p>
          </li>
        ))}
      </ul>
      <p className="text-muted-foreground max-w-md text-sm text-balance">
        Send a photo, a voice note, a video or a paragraph of text, here or on
        WhatsApp. Sherlock shows the evidence behind every answer, and when he
        isn&apos;t sure, he tilts his head instead of guessing.
      </p>
    </aside>
  </main>
);

export default AuthLayout;
