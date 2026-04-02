package curvegpu_test

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type phase5Vectors struct {
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

func TestBN254FrNTTAgainstGnarkCrypto(t *testing.T) {
	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		t.Skipf("WebGPU device unavailable: %v", err)
	}
	defer deviceSet.Close()

	kernel, err := bn254gpu.NewFrNTTKernel(deviceSet.Device)
	if err != nil {
		t.Fatalf("NewFrNTTKernel: %v", err)
	}
	defer kernel.Close()

	vectors, err := loadPhase5Vectors()
	if err != nil {
		t.Fatalf("loadPhase5Vectors: %v", err)
	}

	for _, tc := range vectors.NTTCases {
		tc := tc
		t.Run(tc.Name, func(t *testing.T) {
			input := mustBatch(tc.InputMontLE)
			stageTwiddles := mustStageTwiddles(tc.StageTwiddlesLE)
			inverseStageTwiddles := mustStageTwiddles(tc.InverseStageTwiddlesLE)
			scale := mustU32x8Phase5(tc.InverseScaleLE)

			forward, err := kernel.ForwardDIT(input, stageTwiddles)
			if err != nil {
				t.Fatalf("ForwardDIT: %v", err)
			}
			mustEqualHexBatch(t, "forward", forward, tc.ForwardExpectedLE)

			inverse, err := kernel.InverseDIT(forward, inverseStageTwiddles, scale)
			if err != nil {
				t.Fatalf("InverseDIT: %v", err)
			}
			mustEqualHexBatch(t, "inverse", inverse, tc.InverseExpectedLE)
		})
	}
}

func loadPhase5Vectors() (phase5Vectors, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return phase5Vectors{}, os.ErrNotExist
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "testdata", "vectors", "fr", "bn254_phase5_ntt.json")
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return phase5Vectors{}, err
	}
	var out phase5Vectors
	if err := json.Unmarshal(data, &out); err != nil {
		return phase5Vectors{}, err
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
		out[i] = mustU32x8Phase5(raw[i])
	}
	return out
}

func mustU32x8Phase5(raw string) curvegpu.U32x8 {
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

func mustEqualHexBatch(t *testing.T, name string, got []curvegpu.U32x8, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		gotHex := limbsToHexPhase5(got[i])
		if gotHex != want[i] {
			t.Fatalf("%s mismatch at index %d: got=%s want=%s", name, i, gotHex, want[i])
		}
	}
}

func limbsToHexPhase5(in curvegpu.U32x8) string {
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
