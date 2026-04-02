package main

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"math/bits"
	"math/rand"
	"os"
	"path/filepath"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type phase4Vectors struct {
	VectorCases []vectorCase `json:"vector_cases"`
}

type vectorCase struct {
	Name               string   `json:"name"`
	RegularInputs      []string `json:"regular_inputs_le"`
	MontInputs         []string `json:"mont_inputs_le"`
	MontFactors        []string `json:"mont_factors_le"`
	AddExpected        []string `json:"add_expected_le"`
	SubExpected        []string `json:"sub_expected_le"`
	MulExpected        []string `json:"mul_expected_le"`
	ToMontExpected     []string `json:"to_mont_expected_le"`
	FromMontExpected   []string `json:"from_mont_expected_le"`
	BitReverseExpected []string `json:"bit_reverse_expected_le"`
}

func main() {
	outPath := flag.String("out", filepath.FromSlash("testdata/vectors/fr/bn254_phase4_vector_ops.json"), "output path")
	flag.Parse()

	vectors := phase4Vectors{
		VectorCases: []vectorCase{
			buildVectorCase("n8_random", 8, rand.New(rand.NewSource(2026040201))),
			buildVectorCase("n16_random", 16, rand.New(rand.NewSource(2026040202))),
		},
	}

	if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
		panic(err)
	}
	data, err := json.MarshalIndent(vectors, "", "  ")
	if err != nil {
		panic(err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(*outPath, data, 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("wrote %s\n", *outPath)
}

func buildVectorCase(name string, size int, rng *rand.Rand) vectorCase {
	out := vectorCase{
		Name:               name,
		RegularInputs:      make([]string, size),
		MontInputs:         make([]string, size),
		MontFactors:        make([]string, size),
		AddExpected:        make([]string, size),
		SubExpected:        make([]string, size),
		MulExpected:        make([]string, size),
		ToMontExpected:     make([]string, size),
		FromMontExpected:   make([]string, size),
		BitReverseExpected: make([]string, size),
	}

	for i := 0; i < size; i++ {
		aRegular := randomFieldBigInt(rng)
		bRegular := randomFieldBigInt(rng)
		var aMont, bMont gnarkfr.Element
		aMont.SetBigInt(aRegular)
		bMont.SetBigInt(bRegular)

		var add gnarkfr.Element
		add.Add(&aMont, &bMont)

		var sub gnarkfr.Element
		sub.Sub(&aMont, &bMont)

		var mul gnarkfr.Element
		mul.Mul(&aMont, &bMont)

		out.RegularInputs[i] = bigIntToLittleEndianHex(aRegular)
		out.MontInputs[i] = elementToHex(aMont)
		out.MontFactors[i] = elementToHex(bMont)
		out.AddExpected[i] = elementToHex(add)
		out.SubExpected[i] = elementToHex(sub)
		out.MulExpected[i] = elementToHex(mul)
		out.ToMontExpected[i] = elementToHex(aMont)
		out.FromMontExpected[i] = bigIntToLittleEndianHex(aRegular)
	}

	logCount := bits.Len(uint(size)) - 1
	for i := 0; i < size; i++ {
		j := int(bits.Reverse64(uint64(i)) >> (64 - logCount))
		out.BitReverseExpected[i] = out.MontInputs[j]
	}

	return out
}

func randomFieldBigInt(rng *rand.Rand) *big.Int {
	modulus := gnarkfr.Modulus()
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), modulus)
}

func elementToHex(in gnarkfr.Element) string {
	var bytes [32]byte
	for i, word := range [4]uint64(in) {
		binary.LittleEndian.PutUint64(bytes[i*8:], word)
	}
	return hex.EncodeToString(bytes[:])
}

func bigIntToLittleEndianHex(in *big.Int) string {
	bytes := in.FillBytes(make([]byte, 32))
	for i, j := 0, len(bytes)-1; i < j; i, j = i+1, j-1 {
		bytes[i], bytes[j] = bytes[j], bytes[i]
	}
	return hex.EncodeToString(bytes)
}
