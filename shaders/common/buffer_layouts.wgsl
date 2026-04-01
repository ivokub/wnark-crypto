// Shared buffer layout notes for host and shader code.
//
// 4-word host fields map to 8 x u32 limbs in little-endian order.
// 6-word host fields map to 12 x u32 limbs in little-endian order.
//
// Field element storage is kept in Montgomery form unless documented otherwise.

struct U32x8 {
  limbs: array<u32, 8>,
}

struct U32x12 {
  limbs: array<u32, 12>,
}
