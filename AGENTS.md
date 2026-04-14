# Agent instructions

## Build

Always use `make web-build` to build and type-check the web library. Do not use `npx tsc --noEmit` directly.

## Shader bundle

When adding a new WGSL shader file that is referenced by the library (i.e. added to `web/src/curvegpu/curves.ts`), also add its URL path to the `SHADER_PATHS` array in `web/scripts/bundle-shaders.mjs`.
