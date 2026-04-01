package curvegpu_test

import (
	"encoding/binary"
	"math/big"
	"math/rand"
	"testing"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

const differentialBatchSize = 64

func TestBN254FrDifferentialAgainstGnarkCrypto(t *testing.T) {
	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		t.Skipf("WebGPU device unavailable: %v", err)
	}
	defer deviceSet.Close()

	kernel, err := bn254gpu.NewFrKernel(deviceSet.Device)
	if err != nil {
		t.Fatalf("NewFrKernel: %v", err)
	}
	defer kernel.Close()

	rng := rand.New(rand.NewSource(1))
	cases := make([]randomCase, differentialBatchSize)
	for i := range cases {
		cases[i] = newRandomCase(rng)
	}

	t.Run("static", func(t *testing.T) {
		zeros := bn254gpu.ZeroBatch(len(cases))
		wantZero := make([]curvegpu.U32x8, len(cases))
		wantOne := make([]curvegpu.U32x8, len(cases))
		wantOneValue := montElementToU32x8(regularBigIntToMont(big.NewInt(1)))
		for i := range wantOne {
			wantOne[i] = wantOneValue
		}

		mustMatchBatch(t, kernel, "zero", bn254gpu.FrOpZero, zeros, zeros, wantZero)
		mustMatchBatch(t, kernel, "one", bn254gpu.FrOpOne, zeros, zeros, wantOne)
	})

	t.Run("copy", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := make([]curvegpu.U32x8, len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			aBatch[i] = montElementToU32x8(tc.aMont)
			bBatch[i] = montElementToU32x8(tc.bMont)
			want[i] = aBatch[i]
		}
		mustMatchBatch(t, kernel, "copy", bn254gpu.FrOpCopy, aBatch, bBatch, want)
	})

	t.Run("equal", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := make([]curvegpu.U32x8, len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		one := montElementToU32x8(regularBigIntToMont(big.NewInt(1)))
		for i, tc := range cases {
			aBatch[i] = montElementToU32x8(tc.aMont)
			bBatch[i] = montElementToU32x8(tc.bMont)
			if tc.aMont.Equal(&tc.bMont) {
				want[i] = one
			}
		}
		mustMatchBatch(t, kernel, "equal", bn254gpu.FrOpEqual, aBatch, bBatch, want)
	})

	t.Run("add", func(t *testing.T) {
		runBinaryOp(t, kernel, cases, "add", bn254gpu.FrOpAdd, func(a, b gnarkfr.Element) gnarkfr.Element {
			var z gnarkfr.Element
			z.Add(&a, &b)
			return z
		})
	})

	t.Run("sub", func(t *testing.T) {
		runBinaryOp(t, kernel, cases, "sub", bn254gpu.FrOpSub, func(a, b gnarkfr.Element) gnarkfr.Element {
			var z gnarkfr.Element
			z.Sub(&a, &b)
			return z
		})
	})

	t.Run("neg", func(t *testing.T) {
		runUnaryOp(t, kernel, cases, "neg", bn254gpu.FrOpNeg, func(a gnarkfr.Element) gnarkfr.Element {
			var z gnarkfr.Element
			z.Neg(&a)
			return z
		})
	})

	t.Run("double", func(t *testing.T) {
		runUnaryOp(t, kernel, cases, "double", bn254gpu.FrOpDouble, func(a gnarkfr.Element) gnarkfr.Element {
			var z gnarkfr.Element
			z.Double(&a)
			return z
		})
	})

	t.Run("mul", func(t *testing.T) {
		runBinaryOp(t, kernel, cases, "mul", bn254gpu.FrOpMul, func(a, b gnarkfr.Element) gnarkfr.Element {
			var z gnarkfr.Element
			z.Mul(&a, &b)
			return z
		})
	})

	t.Run("square", func(t *testing.T) {
		runUnaryOp(t, kernel, cases, "square", bn254gpu.FrOpSquare, func(a gnarkfr.Element) gnarkfr.Element {
			var z gnarkfr.Element
			z.Square(&a)
			return z
		})
	})

	t.Run("to_mont", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := bn254gpu.ZeroBatch(len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			aBatch[i] = regularBigIntToU32x8(tc.aRegular)
			want[i] = montElementToU32x8(tc.aMont)
		}
		mustMatchBatch(t, kernel, "to_mont", bn254gpu.FrOpToMont, aBatch, bBatch, want)
	})

	t.Run("from_mont", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := bn254gpu.ZeroBatch(len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			aBatch[i] = montElementToU32x8(tc.aMont)
			want[i] = regularBigIntToU32x8(tc.aRegular)
		}
		mustMatchBatch(t, kernel, "from_mont", bn254gpu.FrOpFromMont, aBatch, bBatch, want)
	})
}

type randomCase struct {
	aRegular *big.Int
	bRegular *big.Int
	aMont    gnarkfr.Element
	bMont    gnarkfr.Element
}

func newRandomCase(rng *rand.Rand) randomCase {
	aRegular := randomFieldBigInt(rng)
	bRegular := randomFieldBigInt(rng)
	if rng.Intn(8) == 0 {
		bRegular = new(big.Int).Set(aRegular)
	}
	return randomCase{
		aRegular: aRegular,
		bRegular: bRegular,
		aMont:    regularBigIntToMont(aRegular),
		bMont:    regularBigIntToMont(bRegular),
	}
}

func runBinaryOp(t *testing.T, kernel *bn254gpu.FrKernel, cases []randomCase, name string, op bn254gpu.FrOp, fn func(a, b gnarkfr.Element) gnarkfr.Element) {
	t.Helper()
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := make([]curvegpu.U32x8, len(cases))
	want := make([]curvegpu.U32x8, len(cases))
	for i, tc := range cases {
		aBatch[i] = montElementToU32x8(tc.aMont)
		bBatch[i] = montElementToU32x8(tc.bMont)
		want[i] = montElementToU32x8(fn(tc.aMont, tc.bMont))
	}
	mustMatchBatch(t, kernel, name, op, aBatch, bBatch, want)
}

func runUnaryOp(t *testing.T, kernel *bn254gpu.FrKernel, cases []randomCase, name string, op bn254gpu.FrOp, fn func(a gnarkfr.Element) gnarkfr.Element) {
	t.Helper()
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := bn254gpu.ZeroBatch(len(cases))
	want := make([]curvegpu.U32x8, len(cases))
	for i, tc := range cases {
		aBatch[i] = montElementToU32x8(tc.aMont)
		want[i] = montElementToU32x8(fn(tc.aMont))
	}
	mustMatchBatch(t, kernel, name, op, aBatch, bBatch, want)
}

func mustMatchBatch(t *testing.T, kernel *bn254gpu.FrKernel, name string, op bn254gpu.FrOp, aBatch, bBatch, want []curvegpu.U32x8) {
	t.Helper()
	got, err := kernel.Run(op, aBatch, bBatch)
	if err != nil {
		t.Fatalf("%s kernel.Run: %v", name, err)
	}
	if len(got) != len(want) {
		t.Fatalf("%s result length mismatch: got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s mismatch at index %d: got=%x want=%x", name, i, got[i], want[i])
		}
	}
}

func randomFieldBigInt(rng *rand.Rand) *big.Int {
	modulus := gnarkfr.Modulus()
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), modulus)
}

func regularBigIntToMont(v *big.Int) gnarkfr.Element {
	var z gnarkfr.Element
	z.SetBigInt(v)
	return z
}

func montElementToU32x8(v gnarkfr.Element) curvegpu.U32x8 {
	return curvegpu.SplitWords4([4]uint64(v))
}

func regularBigIntToU32x8(v *big.Int) curvegpu.U32x8 {
	var out curvegpu.U32x8
	bytes := v.FillBytes(make([]byte, 32))
	for i := 0; i < 32/4; i++ {
		base := len(bytes) - ((i + 1) * 4)
		out[i] = binary.BigEndian.Uint32(bytes[base : base+4])
	}
	return out
}
