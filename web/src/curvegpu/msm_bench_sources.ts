import { fetchBytes, fetchJSON } from "./browser_utils.js";

export type BaseLoadResult<TBases> = {
  bases: TBases;
  prepMs: number;
};

export type BaseSourceInitResult<TContext> = {
  context: TContext;
  postMetricLines?: string[];
};

export type BaseSourceProvider<TBases, TContext> = {
  init: () => Promise<BaseSourceInitResult<TContext>>;
  loadBases: (args: { context: TContext; size: number }) => Promise<BaseLoadResult<TBases>>;
};

export type FixtureMetadata = {
  count: number;
  point_bytes: number;
  format: string;
};

export type ByteBaseSource = "fixture" | "server";

export type PreferredByteBaseSource = ByteBaseSource | "generated";

export type ByteBaseSourceContext = {
  baseSource: ByteBaseSource;
  baseSeed: number;
  baseFixture: Uint8Array | null;
  fixtureMeta: FixtureMetadata | null;
};

export type PreferredByteBaseSourceContext = {
  baseSource: PreferredByteBaseSource | "auto";
  baseSeed: number;
  baseFixture: Uint8Array | null;
  fixtureMeta: FixtureMetadata | null;
};

function slicePointByteFixture(fixture: Uint8Array, pointBytes: number, count: number): Uint8Array {
  const byteLength = count * pointBytes;
  if (fixture.byteLength < byteLength) {
    throw new Error(`fixture has ${Math.floor(fixture.byteLength / pointBytes)} points, need ${count}`);
  }
  return fixture.slice(0, byteLength);
}

export function createGeneratedBaseSource<TBases>(options: {
  loadBases: (size: number) => Promise<TBases>;
}): BaseSourceProvider<TBases, null> {
  return {
    init: async () => ({ context: null }),
    loadBases: async ({ size }) => {
      const prepStart = performance.now();
      const bases = await options.loadBases(size);
      return { bases, prepMs: performance.now() - prepStart };
    },
  };
}

export function createFixtureOrServerByteBaseSource(options: {
  locationSearch: string;
  pointBytes: number;
  fixtureJSONPath: string;
  fixtureBinPath: string;
  serverBinPath: string;
  defaultSource?: ByteBaseSource;
  defaultSeed?: number;
}): BaseSourceProvider<Uint8Array, ByteBaseSourceContext> {
  const params = new URLSearchParams(options.locationSearch);
  const selectedSource = ((): ByteBaseSource => {
    const value = params.get("base-source") ?? params.get("baseSource") ?? options.defaultSource ?? "fixture";
    return value === "server" ? "server" : "fixture";
  })();
  const selectedSeed = ((): number => {
    const raw = params.get("seed");
    if (!raw) {
      return options.defaultSeed ?? 1;
    }
    const parsed = Number.parseInt(raw, 10);
    return Number.isInteger(parsed) ? parsed : (options.defaultSeed ?? 1);
  })();

  return {
    init: async () => {
      let baseFixture: Uint8Array | null = null;
      let fixtureMeta: FixtureMetadata | null = null;
      let fixtureLoadMs = 0;
      if (selectedSource === "fixture") {
        [fixtureMeta, baseFixture] = await Promise.all([
          fetchJSON<FixtureMetadata>(options.fixtureJSONPath),
          (async () => {
            const start = performance.now();
            const bytes = await fetchBytes(options.fixtureBinPath);
            fixtureLoadMs = performance.now() - start;
            return bytes;
          })(),
        ]);
        if (fixtureMeta.point_bytes !== options.pointBytes) {
          throw new Error(`unexpected fixture point size: ${fixtureMeta.point_bytes}`);
        }
        if (baseFixture.byteLength !== fixtureMeta.count * fixtureMeta.point_bytes) {
          throw new Error(
            `fixture length mismatch: got ${baseFixture.byteLength}, want ${fixtureMeta.count * fixtureMeta.point_bytes}`,
          );
        }
      }
      return {
        context: {
          baseSource: selectedSource,
          baseSeed: selectedSeed,
          baseFixture,
          fixtureMeta,
        },
        postMetricLines:
          selectedSource === "fixture"
            ? [`fixture_load_ms = ${fixtureLoadMs.toFixed(3)}`]
            : [`base_source = server`, `base_seed = ${selectedSeed}`],
      };
    },
    loadBases: async ({ context, size }) => {
      const prepStart = performance.now();
      let bases: Uint8Array;
      if (context.baseSource === "fixture") {
        if (!context.baseFixture || !context.fixtureMeta) {
          throw new Error("fixture source selected but fixture was not loaded");
        }
        bases = slicePointByteFixture(context.baseFixture, options.pointBytes, size);
      } else {
        bases = await fetchBytes(`${options.serverBinPath}?count=${size}&seed=${context.baseSeed}`);
        if (bases.byteLength !== size * options.pointBytes) {
          throw new Error(`server base length mismatch: got ${bases.byteLength}, want ${size * options.pointBytes}`);
        }
      }
      return { bases, prepMs: performance.now() - prepStart };
    },
  };
}

function isFetchFailure(error: unknown): boolean {
  return error instanceof Error;
}

export function createPreferredByteBaseSource(options: {
  locationSearch: string;
  pointBytes: number;
  fixtureJSONPath?: string;
  fixtureBinPath?: string;
  serverBinPath?: string;
  defaultSeed?: number;
  generatedLoadBases?: (size: number) => Promise<Uint8Array>;
}): BaseSourceProvider<Uint8Array, PreferredByteBaseSourceContext> {
  const params = new URLSearchParams(options.locationSearch);
  const explicitSourceRaw = params.get("base-source") ?? params.get("baseSource");
  const explicitSource: PreferredByteBaseSource | null =
    explicitSourceRaw === "fixture" || explicitSourceRaw === "server" || explicitSourceRaw === "generated"
      ? explicitSourceRaw
      : null;
  const selectedSeed = ((): number => {
    const raw = params.get("seed");
    if (!raw) {
      return options.defaultSeed ?? 1;
    }
    const parsed = Number.parseInt(raw, 10);
    return Number.isInteger(parsed) ? parsed : (options.defaultSeed ?? 1);
  })();

  async function tryLoadFixture(): Promise<{ fixtureMeta: FixtureMetadata; baseFixture: Uint8Array; fixtureLoadMs: number } | null> {
    if (!options.fixtureJSONPath || !options.fixtureBinPath) {
      return null;
    }
    const start = performance.now();
    const [fixtureMeta, baseFixture] = await Promise.all([
      fetchJSON<FixtureMetadata>(options.fixtureJSONPath),
      fetchBytes(options.fixtureBinPath),
    ]);
    const fixtureLoadMs = performance.now() - start;
    if (fixtureMeta.point_bytes !== options.pointBytes) {
      throw new Error(`unexpected fixture point size: ${fixtureMeta.point_bytes}`);
    }
    if (baseFixture.byteLength !== fixtureMeta.count * fixtureMeta.point_bytes) {
      throw new Error(
        `fixture length mismatch: got ${baseFixture.byteLength}, want ${fixtureMeta.count * fixtureMeta.point_bytes}`,
      );
    }
    return { fixtureMeta, baseFixture, fixtureLoadMs };
  }

  return {
    init: async () => {
      let fixtureMeta: FixtureMetadata | null = null;
      let baseFixture: Uint8Array | null = null;
      let fixtureLoadMs: number | null = null;

      const tryFixtureFirst = explicitSource === null || explicitSource === "fixture";
      if (tryFixtureFirst) {
        try {
          const loaded = await tryLoadFixture();
          if (loaded) {
            fixtureMeta = loaded.fixtureMeta;
            baseFixture = loaded.baseFixture;
            fixtureLoadMs = loaded.fixtureLoadMs;
          }
        } catch (error) {
          if (explicitSource === "fixture" || !isFetchFailure(error)) {
            throw error;
          }
        }
      }

      const postMetricLines: string[] = [];
      if (fixtureLoadMs !== null) {
        postMetricLines.push("base_source = fixture");
        postMetricLines.push(`fixture_load_ms = ${fixtureLoadMs.toFixed(3)}`);
      } else if (explicitSource !== null) {
        postMetricLines.push(`base_source = ${explicitSource}`);
        if (explicitSource === "server") {
          postMetricLines.push(`base_seed = ${selectedSeed}`);
        }
      } else {
        postMetricLines.push("base_source = auto");
        if (options.serverBinPath) {
          postMetricLines.push(`base_seed = ${selectedSeed}`);
        }
      }

      return {
        context: {
          baseSource: explicitSource ?? "auto",
          baseSeed: selectedSeed,
          baseFixture,
          fixtureMeta,
        },
        postMetricLines,
      };
    },
    loadBases: async ({ context, size }) => {
      const prepStart = performance.now();

      const useFixture =
        (context.baseSource === "fixture" || context.baseSource === "auto") &&
        context.baseFixture &&
        context.baseFixture.byteLength >= size * options.pointBytes;
      if (useFixture) {
        return {
          bases: slicePointByteFixture(context.baseFixture as Uint8Array, options.pointBytes, size),
          prepMs: performance.now() - prepStart,
        };
      }

      const tryServer = async (): Promise<Uint8Array> => {
        if (!options.serverBinPath) {
          throw new Error("server base source is not configured");
        }
        const bases = await fetchBytes(`${options.serverBinPath}?count=${size}&seed=${context.baseSeed}`);
        if (bases.byteLength !== size * options.pointBytes) {
          throw new Error(`server base length mismatch: got ${bases.byteLength}, want ${size * options.pointBytes}`);
        }
        return bases;
      };

      if (context.baseSource === "server") {
        return { bases: await tryServer(), prepMs: performance.now() - prepStart };
      }
      if (context.baseSource === "generated") {
        if (!options.generatedLoadBases) {
          throw new Error("generated base source is not configured");
        }
        return { bases: await options.generatedLoadBases(size), prepMs: performance.now() - prepStart };
      }

      if (context.baseSource === "auto" && options.serverBinPath) {
        try {
          return { bases: await tryServer(), prepMs: performance.now() - prepStart };
        } catch (error) {
          if (!isFetchFailure(error)) {
            throw error;
          }
        }
      }

      if (options.generatedLoadBases) {
        return { bases: await options.generatedLoadBases(size), prepMs: performance.now() - prepStart };
      }

      if (context.baseFixture) {
        throw new Error(`fixture has ${Math.floor(context.baseFixture.byteLength / options.pointBytes)} points, need ${size}`);
      }
      throw new Error("no available base source");
    },
  };
}
