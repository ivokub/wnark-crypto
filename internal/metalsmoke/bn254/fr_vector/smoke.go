package smoke

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"math/bits"
	"math/rand"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

func Run() error { return run() }

func run() error {
	fmt.Println("=== BN254 fr Phase 4 Vector Metal Smoke ===")
	fmt.Println()

	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		return err
	}
	defer deviceSet.Close()

	frKernel, err := bn254gpu.NewFrKernel(deviceSet.Device)
	if err != nil {
		return err
	}
	defer frKernel.Close()

	vectorKernel, err := bn254gpu.NewFrVectorKernel(deviceSet.Device)
	if err != nil {
		return err
	}
	defer vectorKernel.Close()

	rng := rand.New(rand.NewSource(4))
	size := 16
	cases := make([]sampleCase, size)
	for i := range cases {
		cases[i] = newSampleCase(rng)
	}

	fmt.Printf("1. Adapter: %s\n", deviceSet.Adapter.Info().Name)
	fmt.Printf("2. Vector length: %d\n", size)
	fmt.Printf("3. Elementwise cases: %d\n", len(cases))

	aBatch := make([]curvegpu.U32x8, size)
	bBatch := make([]curvegpu.U32x8, size)
	regularBatch := make([]curvegpu.U32x8, size)
	wantAdd := make([]curvegpu.U32x8, size)
	wantSub := make([]curvegpu.U32x8, size)
	wantMul := make([]curvegpu.U32x8, size)
	wantMont := make([]curvegpu.U32x8, size)
	wantRegular := make([]curvegpu.U32x8, size)
	for i, tc := range cases {
		aBatch[i] = curvegpu.SplitWords4([4]uint64(tc.aMont))
		bBatch[i] = curvegpu.SplitWords4([4]uint64(tc.bMont))
		regularBatch[i] = regularBigIntToU32x8(tc.aRegular)
		wantMont[i] = aBatch[i]
		wantRegular[i] = regularBatch[i]

		var add gnarkfr.Element
		add.Add(&tc.aMont, &tc.bMont)
		wantAdd[i] = curvegpu.SplitWords4([4]uint64(add))

		var sub gnarkfr.Element
		sub.Sub(&tc.aMont, &tc.bMont)
		wantSub[i] = curvegpu.SplitWords4([4]uint64(sub))

		var mul gnarkfr.Element
		mul.Mul(&tc.aMont, &tc.bMont)
		wantMul[i] = curvegpu.SplitWords4([4]uint64(mul))
	}

	if err := verifyUnary("copy", func() ([]curvegpu.U32x8, error) { return frKernel.Copy(aBatch) }, aBatch); err != nil {
		return err
	}
	if err := verifyBinary("add", func() ([]curvegpu.U32x8, error) { return frKernel.Add(aBatch, bBatch) }, wantAdd); err != nil {
		return err
	}
	if err := verifyBinary("sub", func() ([]curvegpu.U32x8, error) { return frKernel.Sub(aBatch, bBatch) }, wantSub); err != nil {
		return err
	}
	if err := verifyBinary("mul", func() ([]curvegpu.U32x8, error) { return frKernel.Mul(aBatch, bBatch) }, wantMul); err != nil {
		return err
	}
	if err := verifyUnary("to_mont", func() ([]curvegpu.U32x8, error) { return frKernel.ToMont(regularBatch) }, wantMont); err != nil {
		return err
	}
	if err := verifyUnary("from_mont", func() ([]curvegpu.U32x8, error) { return frKernel.FromMont(aBatch) }, wantRegular); err != nil {
		return err
	}
	if err := verifyBinary("mul_factors", func() ([]curvegpu.U32x8, error) { return vectorKernel.MulFactors(aBatch, bBatch) }, wantMul); err != nil {
		return err
	}

	wantBitReverse := make([]curvegpu.U32x8, size)
	logCount := bits.Len(uint(size)) - 1
	for i := 0; i < size; i++ {
		j := int(bits.Reverse64(uint64(i)) >> (64 - logCount))
		wantBitReverse[i] = aBatch[j]
	}
	if err := verifyUnary("bit_reverse_copy", func() ([]curvegpu.U32x8, error) { return vectorKernel.BitReverseCopy(aBatch) }, wantBitReverse); err != nil {
		return err
	}

	fmt.Println()
	fmt.Println("PASS: BN254 fr Phase 4 vector Metal smoke succeeded")
	return nil
}

type sampleCase struct {
	aRegular *big.Int
	aMont    gnarkfr.Element
	bMont    gnarkfr.Element
}

func newSampleCase(rng *rand.Rand) sampleCase {
	aRegular := randomFieldBigInt(rng)
	bRegular := randomFieldBigInt(rng)
	return sampleCase{
		aRegular: aRegular,
		aMont:    regularBigIntToMont(aRegular),
		bMont:    regularBigIntToMont(bRegular),
	}
}

func verifyUnary(name string, fn func() ([]curvegpu.U32x8, error), want []curvegpu.U32x8) error {
	fmt.Printf("4. %s... ", name)
	got, err := fn()
	if err != nil {
		return fmt.Errorf("%s gpu run: %w", name, err)
	}
	if err := verifyBatch(got, want); err != nil {
		return fmt.Errorf("%s mismatch: %w", name, err)
	}
	fmt.Println("OK")
	return nil
}

func verifyBinary(name string, fn func() ([]curvegpu.U32x8, error), want []curvegpu.U32x8) error {
	return verifyUnary(name, fn, want)
}

func verifyBatch(got, want []curvegpu.U32x8) error {
	if len(got) != len(want) {
		return fmt.Errorf("length got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("index %d got=%x want=%x", i, got[i], want[i])
		}
	}
	return nil
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

func regularBigIntToU32x8(v *big.Int) curvegpu.U32x8 {
	var out curvegpu.U32x8
	bytes := v.FillBytes(make([]byte, 32))
	for i := 0; i < 32/4; i++ {
		base := len(bytes) - ((i + 1) * 4)
		out[i] = binary.BigEndian.Uint32(bytes[base : base+4])
	}
	return out
}
