# CurveGPU Field Layouts

This document defines the canonical host-to-GPU representation used by the
prototype.

## Host representation

Host field elements follow the gnark-crypto convention:

- 4 host words: `[4]uint64`
- 6 host words: `[6]uint64`

Each host word array is interpreted in little-endian word order.

## GPU representation

GPU field elements are represented as little-endian `u32` limbs:

- 4 host words -> `8 x u32`
- 6 host words -> `12 x u32`

For each host word:

- limb `2*i` holds the low 32 bits
- limb `2*i+1` holds the high 32 bits

Example:

```text
word = 0x1122334455667788
low  = 0x55667788
high = 0x11223344
```

So the host word array:

```text
[0x1122334455667788, 0, 0, 0]
```

maps to GPU limbs:

```text
[0x55667788, 0x11223344, 0, 0, 0, 0, 0, 0]
```

## Field state

Unless a specific API says otherwise, field elements stored in GPU buffers are
expected to be in Montgomery form.

## Canonical test serialization

Shared test vectors use little-endian byte serialization derived directly from
the host-word layout.

- 4 host words serialize to 32 bytes
- 6 host words serialize to 48 bytes
- each `uint64` host word is serialized in little-endian byte order
- words are concatenated in little-endian word order

This is an internal transport format for the prototype tests. It is not meant
to replace curve-specific canonical field encodings.

## Test vectors

The shared conversion vectors live under:

```text
testdata/vectors/phase1_word_layout.json
```

These vectors are intentionally generic and validate:

- zero
- one
- representative modulus-minus-one values
- carry-heavy values
- arbitrary random-looking values

They are suitable for both Go and TypeScript host wrappers.
