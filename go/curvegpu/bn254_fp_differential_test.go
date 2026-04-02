package curvegpu_test

import (
	"encoding/binary"
	"math/big"
	"math/rand"
	"testing"

	gnarkfp "github.com/consensys/gnark-crypto/ecc/bn254/fp"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

func TestBN254FpDifferentialAgainstGnarkCrypto(t *testing.T) {
	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		t.Skipf("WebGPU device unavailable: %v", err)
	}
	defer deviceSet.Close()

	kernel, err := bn254gpu.NewFpKernel(deviceSet.Device)
	if err != nil {
		t.Fatalf("NewFpKernel: %v", err)
	}
	defer kernel.Close()

	rng := rand.New(rand.NewSource(2))
	cases := make([]randomFpCase, differentialBatchSize)
	for i := range cases {
		cases[i] = newRandomFpCase(rng)
	}

	t.Run("static", func(t *testing.T) {
		zeros := bn254gpu.ZeroBatch(len(cases))
		wantZero := make([]curvegpu.U32x8, len(cases))
		wantOne := make([]curvegpu.U32x8, len(cases))
		wantOneValue := montFpElementToU32x8(regularBigIntToMontFp(big.NewInt(1)))
		for i := range wantOne {
			wantOne[i] = wantOneValue
		}

		mustMatchFpBatch(t, kernel, "zero", bn254gpu.FpOpZero, zeros, zeros, wantZero)
		mustMatchFpBatch(t, kernel, "one", bn254gpu.FpOpOne, zeros, zeros, wantOne)
	})

	t.Run("copy", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := make([]curvegpu.U32x8, len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			aBatch[i] = montFpElementToU32x8(tc.aMont)
			bBatch[i] = montFpElementToU32x8(tc.bMont)
			want[i] = aBatch[i]
		}
		mustMatchFpBatch(t, kernel, "copy", bn254gpu.FpOpCopy, aBatch, bBatch, want)
	})

	t.Run("equal", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := make([]curvegpu.U32x8, len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		one := montFpElementToU32x8(regularBigIntToMontFp(big.NewInt(1)))
		for i, tc := range cases {
			aBatch[i] = montFpElementToU32x8(tc.aMont)
			bBatch[i] = montFpElementToU32x8(tc.bMont)
			if tc.aMont.Equal(&tc.bMont) {
				want[i] = one
			}
		}
		mustMatchFpBatch(t, kernel, "equal", bn254gpu.FpOpEqual, aBatch, bBatch, want)
	})

	t.Run("add", func(t *testing.T) {
		runFpBinaryOp(t, kernel, cases, "add", bn254gpu.FpOpAdd, func(a, b gnarkfp.Element) gnarkfp.Element {
			var z gnarkfp.Element
			z.Add(&a, &b)
			return z
		})
	})

	t.Run("sub", func(t *testing.T) {
		runFpBinaryOp(t, kernel, cases, "sub", bn254gpu.FpOpSub, func(a, b gnarkfp.Element) gnarkfp.Element {
			var z gnarkfp.Element
			z.Sub(&a, &b)
			return z
		})
	})

	t.Run("neg", func(t *testing.T) {
		runFpUnaryOp(t, kernel, cases, "neg", bn254gpu.FpOpNeg, func(a gnarkfp.Element) gnarkfp.Element {
			var z gnarkfp.Element
			z.Neg(&a)
			return z
		})
	})

	t.Run("double", func(t *testing.T) {
		runFpUnaryOp(t, kernel, cases, "double", bn254gpu.FpOpDouble, func(a gnarkfp.Element) gnarkfp.Element {
			var z gnarkfp.Element
			z.Double(&a)
			return z
		})
	})

	t.Run("mul", func(t *testing.T) {
		runFpBinaryOp(t, kernel, cases, "mul", bn254gpu.FpOpMul, func(a, b gnarkfp.Element) gnarkfp.Element {
			var z gnarkfp.Element
			z.Mul(&a, &b)
			return z
		})
	})

	t.Run("square", func(t *testing.T) {
		runFpUnaryOp(t, kernel, cases, "square", bn254gpu.FpOpSquare, func(a gnarkfp.Element) gnarkfp.Element {
			var z gnarkfp.Element
			z.Square(&a)
			return z
		})
	})

	t.Run("to_mont", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := bn254gpu.ZeroBatch(len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			aBatch[i] = regularBigIntToU32x8Fp(tc.aRegular)
			want[i] = montFpElementToU32x8(tc.aMont)
		}
		mustMatchFpBatch(t, kernel, "to_mont", bn254gpu.FpOpToMont, aBatch, bBatch, want)
	})

	t.Run("from_mont", func(t *testing.T) {
		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := bn254gpu.ZeroBatch(len(cases))
		want := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			aBatch[i] = montFpElementToU32x8(tc.aMont)
			want[i] = regularBigIntToU32x8Fp(tc.aRegular)
		}
		mustMatchFpBatch(t, kernel, "from_mont", bn254gpu.FpOpFromMont, aBatch, bBatch, want)
	})
}

type randomFpCase struct {
	aRegular *big.Int
	bRegular *big.Int
	aMont    gnarkfp.Element
	bMont    gnarkfp.Element
}

func newRandomFpCase(rng *rand.Rand) randomFpCase {
	aRegular := randomFpBigInt(rng)
	bRegular := randomFpBigInt(rng)
	if rng.Intn(8) == 0 {
		bRegular = new(big.Int).Set(aRegular)
	}
	return randomFpCase{
		aRegular: aRegular,
		bRegular: bRegular,
		aMont:    regularBigIntToMontFp(aRegular),
		bMont:    regularBigIntToMontFp(bRegular),
	}
}

func runFpBinaryOp(t *testing.T, kernel *bn254gpu.FpKernel, cases []randomFpCase, name string, op bn254gpu.FpOp, fn func(a, b gnarkfp.Element) gnarkfp.Element) {
	t.Helper()
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := make([]curvegpu.U32x8, len(cases))
	want := make([]curvegpu.U32x8, len(cases))
	for i, tc := range cases {
		aBatch[i] = montFpElementToU32x8(tc.aMont)
		bBatch[i] = montFpElementToU32x8(tc.bMont)
		want[i] = montFpElementToU32x8(fn(tc.aMont, tc.bMont))
	}
	mustMatchFpBatch(t, kernel, name, op, aBatch, bBatch, want)
}

func runFpUnaryOp(t *testing.T, kernel *bn254gpu.FpKernel, cases []randomFpCase, name string, op bn254gpu.FpOp, fn func(a gnarkfp.Element) gnarkfp.Element) {
	t.Helper()
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := bn254gpu.ZeroBatch(len(cases))
	want := make([]curvegpu.U32x8, len(cases))
	for i, tc := range cases {
		aBatch[i] = montFpElementToU32x8(tc.aMont)
		want[i] = montFpElementToU32x8(fn(tc.aMont))
	}
	mustMatchFpBatch(t, kernel, name, op, aBatch, bBatch, want)
}

func mustMatchFpBatch(t *testing.T, kernel *bn254gpu.FpKernel, name string, op bn254gpu.FpOp, aBatch, bBatch, want []curvegpu.U32x8) {
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

func randomFpBigInt(rng *rand.Rand) *big.Int {
	modulus := gnarkfp.Modulus()
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), modulus)
}

func regularBigIntToMontFp(v *big.Int) gnarkfp.Element {
	var z gnarkfp.Element
	z.SetBigInt(v)
	return z
}

func montFpElementToU32x8(v gnarkfp.Element) curvegpu.U32x8 {
	return curvegpu.SplitWords4([4]uint64(v))
}

func regularBigIntToU32x8Fp(v *big.Int) curvegpu.U32x8 {
	var out curvegpu.U32x8
	bytes := v.FillBytes(make([]byte, 32))
	for i := 0; i < 32/4; i++ {
		base := len(bytes) - ((i + 1) * 4)
		out[i] = binary.BigEndian.Uint32(bytes[base : base+4])
	}
	return out
}
