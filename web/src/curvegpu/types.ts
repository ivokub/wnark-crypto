import { GENERATED_SHAPES } from "./curve_shapes_gen.js";

export type CurveID = "bn254" | "bls12_381" | "bls12_377";
export type FieldID = "fr" | "fp";

export interface FieldShape {
  curve: CurveID;
  field: FieldID;
  hostWords: 4 | 6;
  gpuLimbs: 8 | 12;
  byteSize: 32 | 48;
}

export type U32x8 = Uint32Array & { length: 8 };
export type U32x12 = Uint32Array & { length: 12 };

export function shapeFor(curve: CurveID, field: FieldID): FieldShape {
  const fields = GENERATED_SHAPES[curve];
  if (!fields) {
    throw new Error(`unsupported curve ${curve}`);
  }
  const shape = fields[field];
  if (!shape) {
    throw new Error(`unsupported field ${field} for curve ${curve}`);
  }
  return shape;
}
