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

	"github.com/ivokub/wnark-crypto/internal/metal"
	bls12381gpu "github.com/ivokub/wnark-crypto/internal/metal/bls12_381"
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

type g1Case struct {
	Name                string      `json:"name"`
	PAffine             affinePoint `json:"p_affine"`
	QAffine             affinePoint `json:"q_affine"`
	PJacobian           jacPoint    `json:"p_jacobian"`
	PAffineOutput       jacPoint    `json:"p_affine_output"`
	NegPJacobian        jacPoint    `json:"neg_p_jacobian"`
	DoublePJacobian     jacPoint    `json:"double_p_jacobian"`
	AddMixedPPlusQJacob jacPoint    `json:"add_mixed_p_plus_q_jacobian"`
	AffineAddPPlusQ     jacPoint    `json:"affine_add_p_plus_q"`
}

type g1OpsVectors struct {
	PointCases []g1Case `json:"point_cases"`
}

func Run() error { return run() }

func run() error {
	fmt.Println("=== BLS12-381 G1 Ops Metal Smoke ===")
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
	fmt.Printf("2. G1 cases: %d\n", len(vectors.PointCases))

	pAff := make([]bls12381gpu.G1Affine, len(vectors.PointCases))
	qAff := make([]bls12381gpu.G1Affine, len(vectors.PointCases))
	pJac := make([]bls12381gpu.G1Jac, len(vectors.PointCases))
	wantNeg := make([]bls12381gpu.G1Jac, len(vectors.PointCases))
	wantDouble := make([]bls12381gpu.G1Jac, len(vectors.PointCases))
	wantAdd := make([]bls12381gpu.G1Jac, len(vectors.PointCases))
	wantAffine := make([]bls12381gpu.G1Jac, len(vectors.PointCases))
	wantAffineAdd := make([]bls12381gpu.G1Jac, len(vectors.PointCases))
	wantInfinity := make([]bls12381gpu.G1Jac, len(vectors.PointCases))
	for i, tc := range vectors.PointCases {
		pAff[i] = mustAffine(tc.PAffine)
		qAff[i] = mustAffine(tc.QAffine)
		pJac[i] = mustJac(tc.PJacobian)
		wantAffine[i] = mustJac(tc.PAffineOutput)
		wantNeg[i] = mustJac(tc.NegPJacobian)
		wantDouble[i] = mustJac(tc.DoublePJacobian)
		wantAdd[i] = mustJac(tc.AddMixedPPlusQJacob)
		wantAffineAdd[i] = mustJac(tc.AffineAddPPlusQ)
		wantInfinity[i] = bls12381gpu.G1JacInfinity()
	}

	gotCopy, err := kernel.Copy(pJac)
	if err := verify("copy", gotCopy, err, pJac); err != nil {
		return err
	}
	gotInfinity, err := kernel.Infinity(len(vectors.PointCases))
	if err := verify("jac_infinity", gotInfinity, err, wantInfinity); err != nil {
		return err
	}
	gotAffineToJac, err := kernel.AffineToJac(pAff)
	if err := verify("affine_to_jac", gotAffineToJac, err, pJac); err != nil {
		return err
	}
	gotNeg, err := kernel.NegJac(pJac)
	if err := verify("neg_jac", gotNeg, err, wantNeg); err != nil {
		return err
	}
	gotJacToAffine, err := kernel.JacToAffine(pJac)
	if err := verify("jac_to_affine", gotJacToAffine, err, wantAffine); err != nil {
		return err
	}
	gotDouble, err := kernel.DoubleJac(pJac)
	if err := verify("double_jac", gotDouble, err, wantDouble); err != nil {
		return err
	}
	gotAdd, err := kernel.AddMixed(pJac, qAff)
	if err := verify("add_mixed", gotAdd, err, wantAdd); err != nil {
		return err
	}
	gotAffineAdd, err := kernel.AffineAdd(pAff, qAff)
	if err := verify("affine_add", gotAffineAdd, err, wantAffineAdd); err != nil {
		return err
	}

	fmt.Println()
	fmt.Println("PASS: BLS12-381 G1 Ops Metal smoke succeeded")
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
	fmt.Printf("3. %s... OK\n", name)
	return nil
}

func loadVectors() (g1OpsVectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return g1OpsVectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "..", "..", "testdata", "vectors", "g1", "bls12_381_g1_ops.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return g1OpsVectors{}, err
	}
	var out g1OpsVectors
	if err := json.Unmarshal(data, &out); err != nil {
		return g1OpsVectors{}, err
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

func mustU32x12(raw string) metal.U32x12 {
	bytes, err := hex.DecodeString(raw)
	if err != nil {
		panic(err)
	}
	if len(bytes) != 48 {
		panic("invalid element length")
	}
	var out metal.U32x12
	for i := 0; i < 12; i++ {
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
		X: gnarkfp.Element(metal.JoinWords12(in.X)),
		Y: gnarkfp.Element(metal.JoinWords12(in.Y)),
		Z: gnarkfp.Element(metal.JoinWords12(in.Z)),
	}
}
