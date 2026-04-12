package smoke

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkfp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bls12381gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bls12_381"
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
	fmt.Println("=== BLS12-381 G1 Scalar Mul Metal Smoke ===")
	fmt.Println()

	vectors, err := loadVectors()
	if err != nil {
		return err
	}

	deviceSet, err := bls12381gpu.NewHeadlessDevice()
	if err != nil {
		return err
	}
	defer deviceSet.Close()

	kernel, err := bls12381gpu.NewG1Kernel(deviceSet.Device)
	if err != nil {
		return err
	}
	defer kernel.Close()

	fmt.Printf("1. Adapter: %s\n", deviceSet.Adapter.Info().Name)
	fmt.Printf("2. Scalar cases: %d\n", len(vectors.ScalarCases))
	fmt.Printf("3. Base cases: %d\n", len(vectors.BaseCases))

	base := make([]bls12381gpu.G1Affine, len(vectors.ScalarCases))
	scalars := make([]curvegpu.U32x8, len(vectors.ScalarCases))
	want := make([]bls12381gpu.G1Jac, len(vectors.ScalarCases))
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
	baseBatch := make([]bls12381gpu.G1Affine, len(vectors.BaseCases))
	wantBase := make([]bls12381gpu.G1Jac, len(vectors.BaseCases))
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
	fmt.Println("PASS: BLS12-381 G1 Scalar Mul Metal smoke succeeded")
	return nil
}

func verify(name string, got []bls12381gpu.G1Jac, err error, want []bls12381gpu.G1Jac) error {
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
	path := filepath.Join(filepath.Dir(filename), "..", "..", "..", "..", "testdata", "vectors", "g1", "bls12_381_g1_scalar_mul.json")
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

func mustAffine(raw affinePoint) bls12381gpu.G1Affine {
	return bls12381gpu.G1Affine{
		X: mustU32x12(raw.XBytesLE),
		Y: mustU32x12(raw.YBytesLE),
	}
}

func mustJac(raw jacPoint) bls12381gpu.G1Jac {
	return bls12381gpu.G1Jac{
		X: mustU32x12(raw.XBytesLE),
		Y: mustU32x12(raw.YBytesLE),
		Z: mustU32x12(raw.ZBytesLE),
	}
}

func mustU32x12(raw string) curvegpu.U32x12 {
	bytes, err := hex.DecodeString(raw)
	if err != nil {
		panic(err)
	}
	if len(bytes) != 48 {
		panic("invalid element length")
	}
	var out curvegpu.U32x12
	for i := 0; i < 12; i++ {
		base := i * 4
		out[i] = uint32(bytes[base]) |
			(uint32(bytes[base+1]) << 8) |
			(uint32(bytes[base+2]) << 16) |
			(uint32(bytes[base+3]) << 24)
	}
	return out
}

func mustU32x8(raw string) curvegpu.U32x8 {
	bytes, err := hex.DecodeString(raw)
	if err != nil {
		panic(err)
	}
	if len(bytes) != 32 {
		panic("invalid scalar length")
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

func gpuJacToGnark(in bls12381gpu.G1Jac) gnarkbls12381.G1Jac {
	return gnarkbls12381.G1Jac{
		X: gnarkfp.Element(curvegpu.JoinWords12(in.X)),
		Y: gnarkfp.Element(curvegpu.JoinWords12(in.Y)),
		Z: gnarkfp.Element(curvegpu.JoinWords12(in.Z)),
	}
}
