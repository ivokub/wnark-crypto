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
	switch curve {
	case CurveBN254:
		return shapeForBN254(field)
	case CurveBLS12381:
		return shapeForBLS12381(field)
	case CurveBLS12377:
		return shapeForBLS12377(field)
	default:
		return FieldShape{}, fmt.Errorf("unsupported curve %q", curve)
	}
}

func shapeForBN254(field FieldID) (FieldShape, error) {
	switch field {
	case FieldScalar, FieldBase:
		return FieldShape{
			Curve:     CurveBN254,
			Field:     field,
			HostWords: 4,
			GPULimbs:  8,
			ByteSize:  32,
		}, nil
	default:
		return FieldShape{}, fmt.Errorf("unsupported field %q for curve %q", field, CurveBN254)
	}
}

func shapeForBLS12381(field FieldID) (FieldShape, error) {
	switch field {
	case FieldScalar:
		return FieldShape{
			Curve:     CurveBLS12381,
			Field:     field,
			HostWords: 4,
			GPULimbs:  8,
			ByteSize:  32,
		}, nil
	case FieldBase:
		return FieldShape{
			Curve:     CurveBLS12381,
			Field:     field,
			HostWords: 6,
			GPULimbs:  12,
			ByteSize:  48,
		}, nil
	default:
		return FieldShape{}, fmt.Errorf("unsupported field %q for curve %q", field, CurveBLS12381)
	}
}

func shapeForBLS12377(field FieldID) (FieldShape, error) {
	switch field {
	case FieldScalar:
		return FieldShape{
			Curve:     CurveBLS12377,
			Field:     field,
			HostWords: 4,
			GPULimbs:  8,
			ByteSize:  32,
		}, nil
	case FieldBase:
		return FieldShape{
			Curve:     CurveBLS12377,
			Field:     field,
			HostWords: 6,
			GPULimbs:  12,
			ByteSize:  48,
		}, nil
	default:
		return FieldShape{}, fmt.Errorf("unsupported field %q for curve %q", field, CurveBLS12377)
	}
}
