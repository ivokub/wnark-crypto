package curvegpu

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func TestSplitJoinWords4RoundTrip(t *testing.T) {
	in := [4]uint64{
		0,
		1,
		0x1122334455667788,
		0xffffffffffffffff,
	}
	limbs := SplitWords4(in)
	out := JoinWords8(limbs)
	if out != in {
		t.Fatalf("round trip mismatch: got %#v want %#v", out, in)
	}
}

func TestSplitJoinWords6RoundTrip(t *testing.T) {
	in := [6]uint64{
		0,
		1,
		0x1122334455667788,
		0x99aabbccddeeff00,
		0x0123456789abcdef,
		0xffffffffffffffff,
	}
	limbs := SplitWords6(in)
	out := JoinWords12(limbs)
	if out != in {
		t.Fatalf("round trip mismatch: got %#v want %#v", out, in)
	}
}

func TestShapeFor(t *testing.T) {
	tests := []struct {
		curve     CurveID
		field     FieldID
		hostWords int
		gpuLimbs  int
	}{
		{CurveBN254, FieldScalar, 4, 8},
		{CurveBN254, FieldBase, 4, 8},
		{CurveBLS12381, FieldScalar, 4, 8},
		{CurveBLS12381, FieldBase, 6, 12},
		{CurveBLS12377, FieldScalar, 4, 8},
		{CurveBLS12377, FieldBase, 6, 12},
	}
	for _, tc := range tests {
		shape, err := ShapeFor(tc.curve, tc.field)
		if err != nil {
			t.Fatalf("ShapeFor(%q, %q) unexpected error: %v", tc.curve, tc.field, err)
		}
		if shape.HostWords != tc.hostWords || shape.GPULimbs != tc.gpuLimbs {
			t.Fatalf("ShapeFor(%q, %q) got host=%d gpu=%d want host=%d gpu=%d",
				tc.curve, tc.field, shape.HostWords, shape.GPULimbs, tc.hostWords, tc.gpuLimbs)
		}
	}
}

func TestListShaders(t *testing.T) {
	shaders, err := ListShaders()
	if err != nil {
		t.Fatalf("ListShaders() error = %v", err)
	}
	if len(shaders) == 0 {
		t.Fatal("expected shader files to be discoverable")
	}
}

type phase1Vectors struct {
	Word4 []layoutVector `json:"word4"`
	Word6 []layoutVector `json:"word6"`
}

type layoutVector struct {
	Name      string   `json:"name"`
	HostWords []string `json:"host_words"`
	GPULimbs  []string `json:"gpu_limbs"`
	BytesLE   string   `json:"bytes_le"`
}

func TestPhase1WordLayoutVectors(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime caller lookup failed")
	}
	vectorPath := filepath.Join(filepath.Dir(filename), "..", "..", "testdata", "vectors", "phase1_word_layout.json")
	data, err := os.ReadFile(vectorPath)
	if err != nil {
		t.Fatalf("read vectors: %v", err)
	}

	var vectors phase1Vectors
	if err := json.Unmarshal(data, &vectors); err != nil {
		t.Fatalf("unmarshal vectors: %v", err)
	}

	for _, tc := range vectors.Word4 {
		hostWords := parseUint64Words(t, tc.HostWords)
		if len(hostWords) != 4 {
			t.Fatalf("%s: expected 4 host words, got %d", tc.Name, len(hostWords))
		}
		var fixed [4]uint64
		copy(fixed[:], hostWords)
		gotLimbs := SplitWords4(fixed)
		wantLimbs := parseUint32Limbs(t, tc.GPULimbs)
		if !reflect.DeepEqual(gotLimbs[:], wantLimbs) {
			t.Fatalf("%s: gpu limbs mismatch got=%#v want=%#v", tc.Name, gotLimbs, wantLimbs)
		}
		wantBytes := parseBytesLE(t, tc.BytesLE)
		gotBytes := wordsToBytesLE(hostWords)
		if !reflect.DeepEqual(gotBytes, wantBytes) {
			t.Fatalf("%s: bytes mismatch got=%x want=%x", tc.Name, gotBytes, wantBytes)
		}
		roundTrip := JoinWords8(gotLimbs)
		if roundTrip != fixed {
			t.Fatalf("%s: round trip mismatch got=%#v want=%#v", tc.Name, roundTrip, fixed)
		}
	}

	for _, tc := range vectors.Word6 {
		hostWords := parseUint64Words(t, tc.HostWords)
		if len(hostWords) != 6 {
			t.Fatalf("%s: expected 6 host words, got %d", tc.Name, len(hostWords))
		}
		var fixed [6]uint64
		copy(fixed[:], hostWords)
		gotLimbs := SplitWords6(fixed)
		wantLimbs := parseUint32Limbs(t, tc.GPULimbs)
		if !reflect.DeepEqual(gotLimbs[:], wantLimbs) {
			t.Fatalf("%s: gpu limbs mismatch got=%#v want=%#v", tc.Name, gotLimbs, wantLimbs)
		}
		wantBytes := parseBytesLE(t, tc.BytesLE)
		gotBytes := wordsToBytesLE(hostWords)
		if !reflect.DeepEqual(gotBytes, wantBytes) {
			t.Fatalf("%s: bytes mismatch got=%x want=%x", tc.Name, gotBytes, wantBytes)
		}
		roundTrip := JoinWords12(gotLimbs)
		if roundTrip != fixed {
			t.Fatalf("%s: round trip mismatch got=%#v want=%#v", tc.Name, roundTrip, fixed)
		}
	}
}

func parseUint64Words(t *testing.T, raw []string) []uint64 {
	t.Helper()
	out := make([]uint64, len(raw))
	for i, item := range raw {
		v, err := strconv.ParseUint(strings.TrimPrefix(item, "0x"), 16, 64)
		if err != nil {
			t.Fatalf("parse uint64 word %q: %v", item, err)
		}
		out[i] = v
	}
	return out
}

func parseUint32Limbs(t *testing.T, raw []string) []uint32 {
	t.Helper()
	out := make([]uint32, len(raw))
	for i, item := range raw {
		v, err := strconv.ParseUint(strings.TrimPrefix(item, "0x"), 16, 32)
		if err != nil {
			t.Fatalf("parse uint32 limb %q: %v", item, err)
		}
		out[i] = uint32(v)
	}
	return out
}

func parseBytesLE(t *testing.T, raw string) []byte {
	t.Helper()
	out, err := hex.DecodeString(raw)
	if err != nil {
		t.Fatalf("parse bytes %q: %v", raw, err)
	}
	return out
}

func wordsToBytesLE(words []uint64) []byte {
	out := make([]byte, len(words)*8)
	for i, word := range words {
		base := i * 8
		for j := 0; j < 8; j++ {
			out[base+j] = byte(word >> (8 * j))
		}
	}
	return out
}
