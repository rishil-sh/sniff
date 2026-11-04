import * as Cloudflare from "@alchemy.run/cloudflare";
import * as Alchemy from "alchemy";
import * as Config from "effect/Config";
import * as Effect from "effect/Effect";
import "varlock/auto-load";

export default Alchemy.Stack(
  "sniff",
  {
    providers: Cloudflare.providers(),
    state: Cloudflare.state(),
  },
  Effect.gen(function* deploy() {
    const webWorker = yield* Cloudflare.Website.StaticSite("web", {
      bundle: true,
      command: "pnpm run build:cloudflare",
      compatibility: {
        flags: ["nodejs_compat", "global_fetch_strictly_public"],
      },
      cwd: "../../apps/web",
      dev: {
        command: "pnpm run dev:bare",
        url: "http://localhost:3001",
      },
      env: {
        IMAGES: Cloudflare.Images.Images(),
        NEXT_PUBLIC_SERVER_URL: Config.String("NEXT_PUBLIC_SERVER_URL"),
      },
      main: "../../apps/web/.open-next/worker.js",
      // Rebuild shared workspace dependencies until Alchemy has a workspace-aware default memo.
      memo: false,
      outdir: ".open-next/assets",
    });

    return {
      web: webWorker.url,
    };
  })
);
