package curvegpu

import "fmt"

type CurveID string

const (
	CurveBN254    CurveID = "bn254"
	CurveBLS12381 CurveID = "bls12_381"
	CurveBLS12377 CurveID = "bls12_377"
)

type FieldID string

const (
	FieldScalar FieldID = "fr"
	FieldBase   FieldID = "fp"
)

type U32x8 [8]uint32

type U32x12 [12]uint32

type FieldShape struct {
	Curve     CurveID
	Field     FieldID
	HostWords int
	GPULimbs  int
	ByteSize  int
}

func ShapeFor(curve CurveID, field FieldID) (FieldShape, error) {
	fields, ok := generatedShapes[curve]
	if !ok {
		return FieldShape{}, fmt.Errorf("unsupported curve %q", curve)
	}
	shape, ok := fields[field]
	if !ok {
		return FieldShape{}, fmt.Errorf("unsupported field %q for curve %q", field, curve)
	}
	return shape, nil
}
