package smoke

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
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

type scalarMulCase struct {
	Name            string      `json:"name"`
	BaseAffine      affinePoint `json:"base_affine"`
	ScalarBytesLE   string      `json:"scalar_bytes_le"`
	ScalarMulAffine jacPoint    `json:"scalar_mul_affine"`
}

type scalarMulBaseCase struct {
	Name                string   `json:"name"`
	ScalarBytesLE       string   `json:"scalar_bytes_le"`
	ScalarMulBaseAffine jacPoint `json:"scalar_mul_base_affine"`
}

type g1ScalarMulVectors struct {
	GeneratorAffine affinePoint         `json:"generator_affine"`
	OneMontZ        string              `json:"one_mont_z"`
	ScalarCases     []scalarMulCase     `json:"scalar_cases"`
	BaseCases       []scalarMulBaseCase `json:"base_cases"`
}

func Run() error { return run() }

func run() error {
	fmt.Println("=== BN254 G1 Scalar Mul Metal Smoke ===")
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

	kernel, err := bn254gpu.NewG1Kernel(deviceSet.Device)
	if err != nil {
		return err
	}
	defer kernel.Close()

	fmt.Printf("1. Adapter: %s\n", deviceSet.Adapter.Info().Name)
	fmt.Printf("2. Scalar cases: %d\n", len(vectors.ScalarCases))
	fmt.Printf("3. Base cases: %d\n", len(vectors.BaseCases))

	base := make([]bn254gpu.G1Affine, len(vectors.ScalarCases))
	scalars := make([]curvegpu.U32x8, len(vectors.ScalarCases))
	want := make([]bn254gpu.G1Jac, len(vectors.ScalarCases))
	for i, tc := range vectors.ScalarCases {
		base[i] = mustAffine(tc.BaseAffine)
		scalars[i] = mustU32x8(tc.ScalarBytesLE)
		want[i] = mustJac(tc.ScalarMulAffine)
	}
	got, err := kernel.ScalarMulAffine(base, scalars)
	if err := verify("scalar_mul_affine", got, err, want); err != nil {
		return err
	}

	generator := mustAffine(vectors.GeneratorAffine)
	baseScalars := make([]curvegpu.U32x8, len(vectors.BaseCases))
	baseBatch := make([]bn254gpu.G1Affine, len(vectors.BaseCases))
	wantBase := make([]bn254gpu.G1Jac, len(vectors.BaseCases))
	for i, tc := range vectors.BaseCases {
		baseScalars[i] = mustU32x8(tc.ScalarBytesLE)
		baseBatch[i] = generator
		wantBase[i] = mustJac(tc.ScalarMulBaseAffine)
	}
	gotBase, err := kernel.ScalarMulAffine(baseBatch, baseScalars)
	if err := verify("scalar_mul_base_affine", gotBase, err, wantBase); err != nil {
		return err
	}

	fmt.Println()
	fmt.Println("PASS: BN254 G1 Scalar Mul Metal smoke succeeded")
	return nil
}

func verify(name string, got []bn254gpu.G1Jac, err error, want []bn254gpu.G1Jac) error {
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if len(got) != len(want) {
		return fmt.Errorf("%s length mismatch: got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("%s mismatch at index %d: got=%+v want=%+v", name, i, got[i], want[i])
		}
		if got[i].IsInfinity() {
			continue
		}
		p := gpuJacToGnark(got[i])
		if !p.IsOnCurve() {
			return fmt.Errorf("%s result at index %d is not on curve", name, i)
		}
	}
	fmt.Printf("4. %s... OK\n", name)
	return nil
}

func loadVectors() (g1ScalarMulVectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return g1ScalarMulVectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "..", "..", "testdata", "vectors", "g1", "bn254_g1_scalar_mul.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return g1ScalarMulVectors{}, err
	}
	var out g1ScalarMulVectors
	if err := json.Unmarshal(data, &out); err != nil {
		return g1ScalarMulVectors{}, err
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
