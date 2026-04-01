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
  if (curve === "bn254") {
    return {
      curve,
      field,
      hostWords: 4,
      gpuLimbs: 8,
      byteSize: 32,
    };
  }
  if (field === "fr") {
    return {
      curve,
      field,
      hostWords: 4,
      gpuLimbs: 8,
      byteSize: 32,
    };
  }
  return {
    curve,
    field,
    hostWords: 6,
    gpuLimbs: 12,
    byteSize: 48,
  };
}
