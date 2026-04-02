package curvegpu_test

import (
	"math/bits"
	"math/rand"
	"testing"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

func TestBN254FrVectorHelpersAgainstGnarkCrypto(t *testing.T) {
	rng := rand.New(rand.NewSource(3))

	t.Run("batched wrappers", func(t *testing.T) {
		_, scalarKernel, _ := newFrVectorTestKernels(t)
		cases := make([]randomCase, 16)
		for i := range cases {
			cases[i] = newRandomCase(rng)
		}

		aBatch := make([]curvegpu.U32x8, len(cases))
		bBatch := make([]curvegpu.U32x8, len(cases))
		wantAdd := make([]curvegpu.U32x8, len(cases))
		wantSub := make([]curvegpu.U32x8, len(cases))
		wantMul := make([]curvegpu.U32x8, len(cases))
		wantCopy := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			aBatch[i] = montElementToU32x8(tc.aMont)
			bBatch[i] = montElementToU32x8(tc.bMont)
			wantCopy[i] = aBatch[i]

			var add gnarkfr.Element
			add.Add(&tc.aMont, &tc.bMont)
			wantAdd[i] = montElementToU32x8(add)

			var sub gnarkfr.Element
			sub.Sub(&tc.aMont, &tc.bMont)
			wantSub[i] = montElementToU32x8(sub)

			var mul gnarkfr.Element
			mul.Mul(&tc.aMont, &tc.bMont)
			wantMul[i] = montElementToU32x8(mul)
		}

		mustMatchUnaryDirect(t, "copy", scalarKernel.Copy, aBatch, wantCopy)
		mustMatchDirect(t, "add", scalarKernel.Add, aBatch, bBatch, wantAdd)
		mustMatchDirect(t, "sub", scalarKernel.Sub, aBatch, bBatch, wantSub)
		mustMatchDirect(t, "mul", scalarKernel.Mul, aBatch, bBatch, wantMul)
	})

	t.Run("batched conversions", func(t *testing.T) {
		_, scalarKernel, _ := newFrVectorTestKernels(t)
		cases := make([]randomCase, 16)
		for i := range cases {
			cases[i] = newRandomCase(rng)
		}
		regularBatch := make([]curvegpu.U32x8, len(cases))
		montBatch := make([]curvegpu.U32x8, len(cases))
		for i, tc := range cases {
			regularBatch[i] = regularBigIntToU32x8(tc.aRegular)
			montBatch[i] = montElementToU32x8(tc.aMont)
		}
		mustMatchUnaryDirect(t, "to_mont", scalarKernel.ToMont, regularBatch, montBatch)
		mustMatchUnaryDirect(t, "from_mont", scalarKernel.FromMont, montBatch, regularBatch)
	})

	t.Run("mul_factors", func(t *testing.T) {
		_, _, vectorKernel := newFrVectorTestKernels(t)
		size := 16
		input := make([]curvegpu.U32x8, size)
		factors := make([]curvegpu.U32x8, size)
		want := make([]curvegpu.U32x8, size)
		for i := 0; i < size; i++ {
			a := newRandomCase(rng).aMont
			b := newRandomCase(rng).aMont
			input[i] = montElementToU32x8(a)
			factors[i] = montElementToU32x8(b)
			var prod gnarkfr.Element
			prod.Mul(&a, &b)
			want[i] = montElementToU32x8(prod)
		}
		got, err := vectorKernel.MulFactors(input, factors)
		if err != nil {
			t.Fatalf("MulFactors: %v", err)
		}
		mustEqualBatch(t, "mul_factors", got, want)
	})

	t.Run("bit_reverse_copy", func(t *testing.T) {
		_, _, vectorKernel := newFrVectorTestKernels(t)
		size := 16
		input := make([]curvegpu.U32x8, size)
		want := make([]curvegpu.U32x8, size)
		for i := 0; i < size; i++ {
			tc := newRandomCase(rng)
			input[i] = montElementToU32x8(tc.aMont)
		}
		logCount := bits.Len(uint(size)) - 1
		for i := 0; i < size; i++ {
			j := int(bits.Reverse64(uint64(i)) >> (64 - logCount))
			want[i] = input[j]
		}
		got, err := vectorKernel.BitReverseCopy(input)
		if err != nil {
			t.Fatalf("BitReverseCopy: %v", err)
		}
		mustEqualBatch(t, "bit_reverse_copy", got, want)
	})
}

func newFrVectorTestKernels(t *testing.T) (*bn254gpu.DeviceSet, *bn254gpu.FrKernel, *bn254gpu.FrVectorKernel) {
	t.Helper()

	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		t.Skipf("WebGPU device unavailable: %v", err)
	}
	t.Cleanup(deviceSet.Close)

	scalarKernel, err := bn254gpu.NewFrKernel(deviceSet.Device)
	if err != nil {
		t.Fatalf("NewFrKernel: %v", err)
	}
	t.Cleanup(scalarKernel.Close)

	vectorKernel, err := bn254gpu.NewFrVectorKernel(deviceSet.Device)
	if err != nil {
		t.Fatalf("NewFrVectorKernel: %v", err)
	}
	t.Cleanup(vectorKernel.Close)

	return deviceSet, scalarKernel, vectorKernel
}

func mustMatchDirect(t *testing.T, name string, fn func([]curvegpu.U32x8, []curvegpu.U32x8) ([]curvegpu.U32x8, error), a, b, want []curvegpu.U32x8) {
	t.Helper()
	got, err := fn(a, b)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	mustEqualBatch(t, name, got, want)
}

func mustMatchUnaryDirect(t *testing.T, name string, fn func([]curvegpu.U32x8) ([]curvegpu.U32x8, error), a, want []curvegpu.U32x8) {
	t.Helper()
	got, err := fn(a)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	mustEqualBatch(t, name, got, want)
}

func mustEqualBatch(t *testing.T, name string, got, want []curvegpu.U32x8) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s mismatch at index %d: got=%x want=%x", name, i, got[i], want[i])
		}
	}
}
