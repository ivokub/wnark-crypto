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

type phase8Vectors struct {
	TermsPerInstance int             `json:"terms_per_instance"`
	MSMCases         []g1MSMCaseJSON `json:"msm_cases"`
	OneMontZ         string          `json:"one_mont_z"`
}

func TestBN254G1MSMKernelAgainstGnarkCrypto(t *testing.T) {
	vectors, err := loadPhase8G1MSMVectors()
	if err != nil {
		t.Fatalf("loadPhase8G1MSMVectors: %v", err)
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
	mustEqualG1Batch(t, "msm_naive", got, want)
	mustBeOnCurveOrInfinity(t, "msm_naive", got)
}

func loadPhase8G1MSMVectors() (phase8Vectors, error) {
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
