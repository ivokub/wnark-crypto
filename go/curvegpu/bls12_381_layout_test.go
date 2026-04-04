package curvegpu_test

import (
	"testing"

	bls381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	bls381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	"github.com/ivokub/wnark-crypto/go/curvegpu"
)

func TestBLS12381FieldShapes(t *testing.T) {
	tests := []struct {
		curve     curvegpu.CurveID
		field     curvegpu.FieldID
		hostWords int
		gpuLimbs  int
		byteSize  int
	}{
		{curvegpu.CurveBLS12381, curvegpu.FieldScalar, 4, 8, 32},
		{curvegpu.CurveBLS12381, curvegpu.FieldBase, 6, 12, 48},
		{curvegpu.CurveBLS12377, curvegpu.FieldScalar, 4, 8, 32},
		{curvegpu.CurveBLS12377, curvegpu.FieldBase, 6, 12, 48},
	}

	for _, tt := range tests {
		shape, err := curvegpu.ShapeFor(tt.curve, tt.field)
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
	limbs := curvegpu.SplitWords4([4]uint64(in))
	out := curvegpu.JoinWords8(limbs)
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
	limbs := curvegpu.SplitWords6([6]uint64(in))
	out := curvegpu.JoinWords12(limbs)
	if out != [6]uint64(in) {
		t.Fatalf("base roundtrip mismatch: got %x want %x", out, [6]uint64(in))
	}
}
