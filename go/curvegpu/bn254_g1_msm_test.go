package curvegpu_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type g1MSMCaseJSON struct {
	Name           string         `json:"name"`
	BasesAffine    []g1AffineJSON `json:"bases_affine"`
	ScalarsBytesLE []string       `json:"scalars_bytes_le"`
	ExpectedAffine g1JacJSON      `json:"expected_affine"`
}

type g1MSMVectors struct {
	TermsPerInstance int             `json:"terms_per_instance"`
	MSMCases         []g1MSMCaseJSON `json:"msm_cases"`
	OneMontZ         string          `json:"one_mont_z"`
}

func TestBN254G1MSMKernelAgainstGnarkCrypto(t *testing.T) {
	vectors, err := loadG1MSMVectors()
	if err != nil {
		t.Fatalf("loadG1MSMVectors: %v", err)
	}

	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		t.Skipf("WebGPU device unavailable: %v", err)
	}
	defer deviceSet.Close()

	kernel, err := bn254gpu.NewG1MSMKernel(deviceSet.Device)
	if err != nil {
		t.Fatalf("NewG1MSMKernel: %v", err)
	}
	defer kernel.Close()

	var bases []bn254gpu.G1Affine
	var scalars []curvegpu.U32x8
	want := make([]bn254gpu.G1Jac, len(vectors.MSMCases))
	for i, tc := range vectors.MSMCases {
		for _, base := range tc.BasesAffine {
			bases = append(bases, mustAffine(base))
		}
		for _, scalar := range tc.ScalarsBytesLE {
			scalars = append(scalars, mustU32x8G1(scalar))
		}
		want[i] = mustJac(tc.ExpectedAffine)
	}

	got, err := kernel.RunAffineNaive(bases, scalars, vectors.TermsPerInstance)
	if err != nil {
		t.Fatalf("RunAffineNaive: %v", err)
	}
	mustEqualG1BatchSemantically(t, "msm_naive", got, want)
	mustBeOnCurveOrInfinity(t, "msm_naive", got)
}

func loadG1MSMVectors() (g1MSMVectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return g1MSMVectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "testdata", "vectors", "g1", "bn254_g1_msm.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return g1MSMVectors{}, err
	}
	var out g1MSMVectors
	if err := json.Unmarshal(data, &out); err != nil {
		return g1MSMVectors{}, err
	}
	return out, nil
}
