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

type g1ScalarCaseJSON struct {
	Name            string       `json:"name"`
	BaseAffine      g1AffineJSON `json:"base_affine"`
	ScalarBytesLE   string       `json:"scalar_bytes_le"`
	ScalarMulAffine g1JacJSON    `json:"scalar_mul_affine"`
}

type g1ScalarBaseCaseJSON struct {
	Name                string    `json:"name"`
	ScalarBytesLE       string    `json:"scalar_bytes_le"`
	ScalarMulBaseAffine g1JacJSON `json:"scalar_mul_base_affine"`
}

type phase7Vectors struct {
	GeneratorAffine g1AffineJSON           `json:"generator_affine"`
	OneMontZ        string                 `json:"one_mont_z"`
	ScalarCases     []g1ScalarCaseJSON     `json:"scalar_cases"`
	BaseCases       []g1ScalarBaseCaseJSON `json:"base_cases"`
}

func TestBN254G1ScalarMulKernelAgainstGnarkCrypto(t *testing.T) {
	vectors, err := loadPhase7G1ScalarVectors()
	if err != nil {
		t.Fatalf("loadPhase7G1ScalarVectors: %v", err)
	}

	t.Run("scalar_mul_affine", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		base := make([]bn254gpu.G1Affine, len(vectors.ScalarCases))
		scalars := make([]curvegpu.U32x8, len(vectors.ScalarCases))
		want := make([]bn254gpu.G1Jac, len(vectors.ScalarCases))
		for i, tc := range vectors.ScalarCases {
			base[i] = mustAffine(tc.BaseAffine)
			scalars[i] = mustU32x8G1(tc.ScalarBytesLE)
			want[i] = mustJac(tc.ScalarMulAffine)
		}
		got, err := kernel.ScalarMulAffine(base, scalars)
		if err != nil {
			t.Fatalf("ScalarMulAffine: %v", err)
		}
		mustEqualG1Batch(t, "scalar_mul_affine", got, want)
		mustBeOnCurveOrInfinity(t, "scalar_mul_affine", got)
	})

	t.Run("scalar_mul_base_affine", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		scalars := make([]curvegpu.U32x8, len(vectors.BaseCases))
		want := make([]bn254gpu.G1Jac, len(vectors.BaseCases))
		base := make([]bn254gpu.G1Affine, len(vectors.BaseCases))
		generator := mustAffine(vectors.GeneratorAffine)
		for i, tc := range vectors.BaseCases {
			scalars[i] = mustU32x8G1(tc.ScalarBytesLE)
			want[i] = mustJac(tc.ScalarMulBaseAffine)
			base[i] = generator
		}
		got, err := kernel.ScalarMulAffine(base, scalars)
		if err != nil {
			t.Fatalf("ScalarMulBaseAffine: %v", err)
		}
		mustEqualG1Batch(t, "scalar_mul_base_affine", got, want)
		mustBeOnCurveOrInfinity(t, "scalar_mul_base_affine", got)
	})
}

func loadPhase7G1ScalarVectors() (phase7Vectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return phase7Vectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "testdata", "vectors", "g1", "bn254_phase7_scalar_mul.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return phase7Vectors{}, err
	}
	var out phase7Vectors
	if err := json.Unmarshal(data, &out); err != nil {
		return phase7Vectors{}, err
	}
	return out, nil
}
