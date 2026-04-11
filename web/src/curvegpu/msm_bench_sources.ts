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

export type ByteBaseSourceContext = {
  baseSource: ByteBaseSource;
  baseSeed: number;
  baseFixture: Uint8Array | null;
  fixtureMeta: FixtureMetadata | null;
};

async function fetchText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchJSON<T>(path: string): Promise<T> {
  return JSON.parse(await fetchText(path)) as T;
}

async function fetchBytes(path: string): Promise<Uint8Array> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

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
