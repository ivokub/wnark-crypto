package smoke

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type frNTTVectors struct {
	NTTCases []nttCase `json:"ntt_cases"`
}

type nttCase struct {
	Name                   string     `json:"name"`
	Size                   int        `json:"size"`
	InputMontLE            []string   `json:"input_mont_le"`
	ForwardExpectedLE      []string   `json:"forward_expected_le"`
	InverseExpectedLE      []string   `json:"inverse_expected_le"`
	StageTwiddlesLE        [][]string `json:"stage_twiddles_le"`
	InverseStageTwiddlesLE [][]string `json:"inverse_stage_twiddles_le"`
	InverseScaleLE         string     `json:"inverse_scale_le"`
}

func Run() error { return run() }

func run() error {
	fmt.Println("=== BN254 fr NTT Metal Smoke ===")
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

	kernel, err := bn254gpu.NewFrNTTKernel(deviceSet.Device)
	if err != nil {
		return err
	}
	defer kernel.Close()

	fmt.Printf("1. Adapter: %s\n", deviceSet.Adapter.Info().Name)
	fmt.Printf("2. NTT cases: %d\n", len(vectors.NTTCases))
	for _, tc := range vectors.NTTCases {
		fmt.Printf("3. %s (n=%d)... ", tc.Name, tc.Size)
		input := mustBatch(tc.InputMontLE)
		stageTwiddles := mustStageTwiddles(tc.StageTwiddlesLE)
		inverseStageTwiddles := mustStageTwiddles(tc.InverseStageTwiddlesLE)
		scale := mustU32x8(tc.InverseScaleLE)

		forward, err := kernel.ForwardDIT(input, stageTwiddles)
		if err != nil {
			return fmt.Errorf("%s forward: %w", tc.Name, err)
		}
		if err := verifyBatch(forward, tc.ForwardExpectedLE); err != nil {
			return fmt.Errorf("%s forward mismatch: %w", tc.Name, err)
		}

		inverse, err := kernel.InverseDIT(forward, inverseStageTwiddles, scale)
		if err != nil {
			return fmt.Errorf("%s inverse: %w", tc.Name, err)
		}
		if err := verifyBatch(inverse, tc.InverseExpectedLE); err != nil {
			return fmt.Errorf("%s inverse mismatch: %w", tc.Name, err)
		}
		fmt.Println("OK")
	}

	fmt.Println()
	fmt.Println("PASS: BN254 fr NTT Metal smoke succeeded")
	return nil
}

func loadVectors() (frNTTVectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return frNTTVectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "..", "..", "testdata", "vectors", "fr", "bn254_fr_ntt.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return frNTTVectors{}, err
	}
	var out frNTTVectors
	if err := json.Unmarshal(data, &out); err != nil {
		return frNTTVectors{}, err
	}
	return out, nil
}

func mustStageTwiddles(raw [][]string) [][]curvegpu.U32x8 {
	out := make([][]curvegpu.U32x8, len(raw))
	for i := range raw {
		out[i] = mustBatch(raw[i])
	}
	return out
}

func mustBatch(raw []string) []curvegpu.U32x8 {
	out := make([]curvegpu.U32x8, len(raw))
	for i := range raw {
		out[i] = mustU32x8(raw[i])
	}
	return out
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
		out[i] = uint32(bytes[i*4]) |
			(uint32(bytes[i*4+1]) << 8) |
			(uint32(bytes[i*4+2]) << 16) |
			(uint32(bytes[i*4+3]) << 24)
	}
	return out
}

func verifyBatch(got []curvegpu.U32x8, want []string) error {
	if len(got) != len(want) {
		return fmt.Errorf("length got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if limbsToHex(got[i]) != want[i] {
			return fmt.Errorf("index %d got=%s want=%s", i, limbsToHex(got[i]), want[i])
		}
	}
	return nil
}

func limbsToHex(in curvegpu.U32x8) string {
	var bytes [32]byte
	for i, limb := range in {
		base := i * 4
		bytes[base+0] = byte(limb)
		bytes[base+1] = byte(limb >> 8)
		bytes[base+2] = byte(limb >> 16)
		bytes[base+3] = byte(limb >> 24)
	}
	return hex.EncodeToString(bytes[:])
}
