package curvegpu_test

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkfp "github.com/consensys/gnark-crypto/ecc/bn254/fp"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type g1AffineJSON struct {
	XBytesLE string `json:"x_bytes_le"`
	YBytesLE string `json:"y_bytes_le"`
}

type g1JacJSON struct {
	XBytesLE string `json:"x_bytes_le"`
	YBytesLE string `json:"y_bytes_le"`
	ZBytesLE string `json:"z_bytes_le"`
}

type g1CaseJSON struct {
	Name                string       `json:"name"`
	PAffine             g1AffineJSON `json:"p_affine"`
	QAffine             g1AffineJSON `json:"q_affine"`
	PJacobian           g1JacJSON    `json:"p_jacobian"`
	PAffineOutput       g1JacJSON    `json:"p_affine_output"`
	NegPJacobian        g1JacJSON    `json:"neg_p_jacobian"`
	DoublePJacobian     g1JacJSON    `json:"double_p_jacobian"`
	AddMixedPPlusQJacob g1JacJSON    `json:"add_mixed_p_plus_q_jacobian"`
	AffineAddPPlusQ     g1JacJSON    `json:"affine_add_p_plus_q"`
}

type phase6Vectors struct {
	PointCases []g1CaseJSON `json:"point_cases"`
}

func TestBN254G1KernelAgainstGnarkCrypto(t *testing.T) {
	vectors, err := loadPhase6G1Vectors()
	if err != nil {
		t.Fatalf("loadPhase6G1Vectors: %v", err)
	}

	pAff := make([]bn254gpu.G1Affine, len(vectors.PointCases))
	qAff := make([]bn254gpu.G1Affine, len(vectors.PointCases))
	pJac := make([]bn254gpu.G1Jac, len(vectors.PointCases))
	wantNeg := make([]bn254gpu.G1Jac, len(vectors.PointCases))
	wantDouble := make([]bn254gpu.G1Jac, len(vectors.PointCases))
	wantAdd := make([]bn254gpu.G1Jac, len(vectors.PointCases))
	wantAffine := make([]bn254gpu.G1Jac, len(vectors.PointCases))
	wantAffineAdd := make([]bn254gpu.G1Jac, len(vectors.PointCases))
	wantInfinity := make([]bn254gpu.G1Jac, len(vectors.PointCases))
	for i, tc := range vectors.PointCases {
		pAff[i] = mustAffine(tc.PAffine)
		qAff[i] = mustAffine(tc.QAffine)
		pJac[i] = mustJac(tc.PJacobian)
		wantAffine[i] = mustJac(tc.PAffineOutput)
		wantNeg[i] = mustJac(tc.NegPJacobian)
		wantDouble[i] = mustJac(tc.DoublePJacobian)
		wantAdd[i] = mustJac(tc.AddMixedPPlusQJacob)
		wantAffineAdd[i] = mustJac(tc.AffineAddPPlusQ)
		wantInfinity[i] = bn254gpu.G1JacInfinity()
	}

	t.Run("copy", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.Copy(pJac)
		if err != nil {
			t.Fatalf("Copy: %v", err)
		}
		mustEqualG1Batch(t, "copy", got, pJac)
	})

	t.Run("jac_infinity", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.Infinity(len(vectors.PointCases))
		if err != nil {
			t.Fatalf("Infinity: %v", err)
		}
		mustEqualG1Batch(t, "jac_infinity", got, wantInfinity)
	})

	t.Run("affine_to_jac", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.AffineToJac(pAff)
		if err != nil {
			t.Fatalf("AffineToJac: %v", err)
		}
		mustEqualG1Batch(t, "affine_to_jac", got, pJac)
		mustBeOnCurve(t, "affine_to_jac", got)
	})

	t.Run("neg_jac", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.NegJac(pJac)
		if err != nil {
			t.Fatalf("NegJac: %v", err)
		}
		mustEqualG1Batch(t, "neg_jac", got, wantNeg)
		mustBeOnCurve(t, "neg_jac", got)
	})

	t.Run("jac_to_affine", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.JacToAffine(pJac)
		if err != nil {
			t.Fatalf("JacToAffine: %v", err)
		}
		mustEqualG1Batch(t, "jac_to_affine", got, wantAffine)
		mustBeOnCurveOrInfinity(t, "jac_to_affine", got)
	})

	t.Run("double_jac", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.DoubleJac(pJac)
		if err != nil {
			t.Fatalf("DoubleJac: %v", err)
		}
		mustEqualG1Batch(t, "double_jac", got, wantDouble)
		mustBeOnCurve(t, "double_jac", got)
	})

	t.Run("add_mixed", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.AddMixed(pJac, qAff)
		if err != nil {
			t.Fatalf("AddMixed: %v", err)
		}
		mustEqualG1Batch(t, "add_mixed", got, wantAdd)
		mustBeOnCurve(t, "add_mixed", got)
	})

	t.Run("affine_add", func(t *testing.T) {
		kernel := newG1TestKernel(t)
		got, err := kernel.AffineAdd(pAff, qAff)
		if err != nil {
			t.Fatalf("AffineAdd: %v", err)
		}
		mustEqualG1Batch(t, "affine_add", got, wantAffineAdd)
		mustBeOnCurveOrInfinity(t, "affine_add", got)
	})
}

func newG1TestKernel(t *testing.T) *bn254gpu.G1Kernel {
	t.Helper()
	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		t.Skipf("WebGPU device unavailable: %v", err)
	}
	t.Cleanup(deviceSet.Close)

	kernel, err := bn254gpu.NewG1Kernel(deviceSet.Device)
	if err != nil {
		t.Fatalf("NewG1Kernel: %v", err)
	}
	t.Cleanup(kernel.Close)
	return kernel
}

func loadPhase6G1Vectors() (phase6Vectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return phase6Vectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "testdata", "vectors", "g1", "bn254_phase6_g1_ops.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return phase6Vectors{}, err
	}
	var out phase6Vectors
	if err := json.Unmarshal(data, &out); err != nil {
		return phase6Vectors{}, err
	}
	return out, nil
}

func mustAffine(raw g1AffineJSON) bn254gpu.G1Affine {
	return bn254gpu.G1Affine{
		X: mustU32x8G1(raw.XBytesLE),
		Y: mustU32x8G1(raw.YBytesLE),
	}
}

func mustJac(raw g1JacJSON) bn254gpu.G1Jac {
	return bn254gpu.G1Jac{
		X: mustU32x8G1(raw.XBytesLE),
		Y: mustU32x8G1(raw.YBytesLE),
		Z: mustU32x8G1(raw.ZBytesLE),
	}
}

func mustU32x8G1(raw string) curvegpu.U32x8 {
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

func mustEqualG1Batch(t *testing.T, name string, got, want []bn254gpu.G1Jac) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s mismatch at index %d: got=%+v want=%+v", name, i, got[i], want[i])
		}
	}
}

func mustEqualG1BatchSemantically(t *testing.T, name string, got, want []bn254gpu.G1Jac) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		gotAff := gpuJacToGnarkAffine(got[i])
		wantAff := gpuJacToGnarkAffine(want[i])
		if !gotAff.Equal(&wantAff) {
			t.Fatalf("%s mismatch at index %d: got=%+v want=%+v", name, i, got[i], want[i])
		}
	}
}

func mustBeOnCurve(t *testing.T, name string, got []bn254gpu.G1Jac) {
	t.Helper()
	for i := range got {
		p := gpuJacToGnark(got[i])
		if !p.IsOnCurve() {
			t.Fatalf("%s result at index %d is not on curve", name, i)
		}
	}
}

func mustBeOnCurveOrInfinity(t *testing.T, name string, got []bn254gpu.G1Jac) {
	t.Helper()
	for i := range got {
		if got[i].IsInfinity() {
			continue
		}
		p := gpuJacToGnark(got[i])
		if !p.IsOnCurve() {
			t.Fatalf("%s result at index %d is not on curve", name, i)
		}
	}
}

func gpuJacToGnark(in bn254gpu.G1Jac) gnarkbn254.G1Jac {
	return gnarkbn254.G1Jac{
		X: limbsToFpElement(in.X),
		Y: limbsToFpElement(in.Y),
		Z: limbsToFpElement(in.Z),
	}
}

func gpuJacToGnarkAffine(in bn254gpu.G1Jac) gnarkbn254.G1Affine {
	jac := gpuJacToGnark(in)
	var aff gnarkbn254.G1Affine
	aff.FromJacobian(&jac)
	return aff
}

func limbsToFpElement(in curvegpu.U32x8) gnarkfp.Element {
	return gnarkfp.Element(curvegpu.JoinWords8(in))
}
