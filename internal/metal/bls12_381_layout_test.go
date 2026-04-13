package metal_test

import (
	"testing"

	bls381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	bls381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	"github.com/ivokub/wnark-crypto/internal/metal"
)

func TestBLS12381FieldShapes(t *testing.T) {
	tests := []struct {
		curve     metal.CurveID
		field     metal.FieldID
		hostWords int
		gpuLimbs  int
		byteSize  int
	}{
		{metal.CurveBLS12381, metal.FieldScalar, 4, 8, 32},
		{metal.CurveBLS12381, metal.FieldBase, 6, 12, 48},
		{metal.CurveBLS12377, metal.FieldScalar, 4, 8, 32},
		{metal.CurveBLS12377, metal.FieldBase, 6, 12, 48},
	}

	for _, tt := range tests {
		shape, err := metal.ShapeFor(tt.curve, tt.field)
		if err != nil {
			t.Fatalf("ShapeFor(%s,%s) failed: %v", tt.curve, tt.field, err)
		}
		if shape.HostWords != tt.hostWords || shape.GPULimbs != tt.gpuLimbs || shape.ByteSize != tt.byteSize {
			t.Fatalf("unexpected shape for %s/%s: %+v", tt.curve, tt.field, shape)
		}
	}
}

func TestBLS12381ScalarRoundTrip(t *testing.T) {
	in := bls381fr.Element{
		0x1122334455667788,
		0x99aabbccddeeff00,
		0x0123456789abcdef,
		0xfedcba9876543210,
	}
	limbs := metal.SplitWords4([4]uint64(in))
	out := metal.JoinWords8(limbs)
	if out != [4]uint64(in) {
		t.Fatalf("scalar roundtrip mismatch: got %x want %x", out, [4]uint64(in))
	}
}

func TestBLS12381BaseRoundTrip(t *testing.T) {
	in := bls381fp.Element{
		0x1122334455667788,
		0x99aabbccddeeff00,
		0x0123456789abcdef,
		0xfedcba9876543210,
		0x0f1e2d3c4b5a6978,
		0x8877665544332211,
	}
	limbs := metal.SplitWords6([6]uint64(in))
	out := metal.JoinWords12(limbs)
	if out != [6]uint64(in) {
		t.Fatalf("base roundtrip mismatch: got %x want %x", out, [6]uint64(in))
	}
}
