package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"

	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkfp "github.com/consensys/gnark-crypto/ecc/bn254/fp"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type affinePoint struct {
	XBytesLE string `json:"x_bytes_le"`
	YBytesLE string `json:"y_bytes_le"`
}

type jacPoint struct {
	XBytesLE string `json:"x_bytes_le"`
	YBytesLE string `json:"y_bytes_le"`
	ZBytesLE string `json:"z_bytes_le"`
}

type msmCase struct {
	Name           string        `json:"name"`
	BasesAffine    []affinePoint `json:"bases_affine"`
	ScalarsBytesLE []string      `json:"scalars_bytes_le"`
	ExpectedAffine jacPoint      `json:"expected_affine"`
}

type phase8Vectors struct {
	TermsPerInstance int       `json:"terms_per_instance"`
	MSMCases         []msmCase `json:"msm_cases"`
	OneMontZ         string    `json:"one_mont_z"`
}

func main() {
	if err := run(); err != nil {
		log.Fatalf("FATAL: %v", err)
	}
}

func run() error {
	fmt.Println("=== BN254 G1 Phase 8 Metal Smoke ===")
	fmt.Println()

	vectors, err := loadVectors()
	if err != nil {
		return err
	}

	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		return err
	}
	defer deviceSet.Close()

	kernel, err := bn254gpu.NewG1MSMKernel(deviceSet.Device)
	if err != nil {
		return err
	}
	defer kernel.Close()

	fmt.Printf("1. Adapter: %s\n", deviceSet.Adapter.Info().Name)
	fmt.Printf("2. Terms per instance: %d\n", vectors.TermsPerInstance)
	fmt.Printf("3. MSM cases: %d\n", len(vectors.MSMCases))

	var bases []bn254gpu.G1Affine
	var scalars []curvegpu.U32x8
	want := make([]bn254gpu.G1Jac, len(vectors.MSMCases))
	for i, tc := range vectors.MSMCases {
		for _, base := range tc.BasesAffine {
			bases = append(bases, mustAffine(base))
		}
		for _, scalar := range tc.ScalarsBytesLE {
			scalars = append(scalars, mustU32x8(scalar))
		}
		want[i] = mustJac(tc.ExpectedAffine)
	}

	got, err := kernel.RunAffineNaive(bases, scalars, vectors.TermsPerInstance)
	if err != nil {
		return err
	}
	if err := verifyBatch("msm_naive_affine", got, want); err != nil {
		return err
	}
	fmt.Println("4. msm_naive_affine... OK")
	fmt.Println()
	fmt.Println("PASS: BN254 G1 Phase 8 Metal smoke succeeded")
	return nil
}

func verifyBatch(label string, got, want []bn254gpu.G1Jac) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s length mismatch: got=%d want=%d", label, len(got), len(want))
	}
	for i := range got {
		gotAff := gpuJacToGnarkAffine(got[i])
		wantAff := gpuJacToGnarkAffine(want[i])
		if !gotAff.Equal(&wantAff) {
			return fmt.Errorf("%s mismatch at index %d: got=%+v want=%+v", label, i, got[i], want[i])
		}
		if got[i].IsInfinity() {
			continue
		}
		p := gpuJacToGnark(got[i])
		if !p.IsOnCurve() {
			return fmt.Errorf("%s result at index %d is not on curve", label, i)
		}
	}
	return nil
}

func loadVectors() (phase8Vectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return phase8Vectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "testdata", "vectors", "g1", "bn254_phase8_msm.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return phase8Vectors{}, err
	}
	var out phase8Vectors
	if err := json.Unmarshal(data, &out); err != nil {
		return phase8Vectors{}, err
	}
	return out, nil
}

func mustAffine(raw affinePoint) bn254gpu.G1Affine {
	return bn254gpu.G1Affine{
		X: mustU32x8(raw.XBytesLE),
		Y: mustU32x8(raw.YBytesLE),
	}
}

func mustJac(raw jacPoint) bn254gpu.G1Jac {
	return bn254gpu.G1Jac{
		X: mustU32x8(raw.XBytesLE),
		Y: mustU32x8(raw.YBytesLE),
		Z: mustU32x8(raw.ZBytesLE),
	}
}

func mustU32x8(raw string) curvegpu.U32x8 {
	bytes, err := hex.DecodeString(raw)
	if err != nil {
		panic(err)
	}
	if len(bytes) != 32 {
		panic("invalid element length")
	}
	var out curvegpu.U32x8
	for i := 0; i < 8; i++ {
		base := i * 4
		out[i] = uint32(bytes[base]) |
			(uint32(bytes[base+1]) << 8) |
			(uint32(bytes[base+2]) << 16) |
			(uint32(bytes[base+3]) << 24)
	}
	return out
}

func gpuJacToGnark(in bn254gpu.G1Jac) gnarkbn254.G1Jac {
	return gnarkbn254.G1Jac{
		X: gnarkfp.Element(curvegpu.JoinWords8(in.X)),
		Y: gnarkfp.Element(curvegpu.JoinWords8(in.Y)),
		Z: gnarkfp.Element(curvegpu.JoinWords8(in.Z)),
	}
}

func gpuJacToGnarkAffine(in bn254gpu.G1Jac) gnarkbn254.G1Affine {
	jac := gpuJacToGnark(in)
	var aff gnarkbn254.G1Affine
	aff.FromJacobian(&jac)
	return aff
}
