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
	gnarkfft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/utils"
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

func main() {
	outPath := flag.String("out", filepath.FromSlash("testdata/vectors/fr/bn254_phase5_ntt.json"), "output path")
	flag.Parse()

	vectors := phase5Vectors{
		NTTCases: []nttCase{
			buildCase("n8_random", 8, rand.New(rand.NewSource(2026040203))),
			buildCase("n16_random", 16, rand.New(rand.NewSource(2026040204))),
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

func buildCase(name string, size int, rng *rand.Rand) nttCase {
	domain := gnarkfft.NewDomain(uint64(size))
	twiddles, err := domain.Twiddles()
	if err != nil {
		panic(err)
	}
	twiddlesInv, err := domain.TwiddlesInv()
	if err != nil {
		panic(err)
	}

	input := make([]gnarkfr.Element, size)
	for i := range input {
		input[i].SetBigInt(randomFieldBigInt(rng))
	}

	forward := make([]gnarkfr.Element, size)
	copy(forward, input)
	utils.BitReverse(forward)
	domain.FFT(forward, gnarkfft.DIT)

	inverse := make([]gnarkfr.Element, size)
	copy(inverse, forward)
	utils.BitReverse(inverse)
	domain.FFTInverse(inverse, gnarkfft.DIT)

	logN := bits.Len(uint(size)) - 1
	stageTwiddles := make([][]string, logN)
	inverseStageTwiddles := make([][]string, logN)
	for stage := 1; stage <= logN; stage++ {
		m := 1 << (stage - 1)
		src := twiddles[logN-stage]
		srcInv := twiddlesInv[logN-stage]
		stageTwiddles[stage-1] = make([]string, m)
		inverseStageTwiddles[stage-1] = make([]string, m)
		for i := 0; i < m; i++ {
			stageTwiddles[stage-1][i] = elementToHex(src[i])
			inverseStageTwiddles[stage-1][i] = elementToHex(srcInv[i])
		}
	}

	out := nttCase{
		Name:                   name,
		Size:                   size,
		InputMontLE:            encodeBatch(input),
		ForwardExpectedLE:      encodeBatch(forward),
		InverseExpectedLE:      encodeBatch(inverse),
		StageTwiddlesLE:        stageTwiddles,
		InverseStageTwiddlesLE: inverseStageTwiddles,
		InverseScaleLE:         elementToHex(domain.CardinalityInv),
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

func encodeBatch(in []gnarkfr.Element) []string {
	out := make([]string, len(in))
	for i := range in {
		out[i] = elementToHex(in[i])
	}
	return out
}

func elementToHex(in gnarkfr.Element) string {
	var bytes [32]byte
	for i, word := range [4]uint64(in) {
		binary.LittleEndian.PutUint64(bytes[i*8:], word)
	}
	return hex.EncodeToString(bytes[:])
}
