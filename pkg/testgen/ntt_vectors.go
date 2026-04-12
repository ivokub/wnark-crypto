package testgen

import (
	"math/big"
	"math/bits"
	"math/rand"

	gnarkbn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
	gnarkfft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/utils"
)

type BN254NTTDomainFileJSON struct {
	Domains []BN254NTTDomainJSON `json:"domains"`
}

type BN254NTTDomainJSON struct {
	LogN              int    `json:"log_n"`
	Size              int    `json:"size"`
	OmegaHex          string `json:"omega_hex"`
	OmegaInvHex       string `json:"omega_inv_hex"`
	CardinalityInvHex string `json:"cardinality_inv_hex"`
}

type BN254NTTVectorsJSON struct {
	NTTCases []BN254NTTCaseJSON `json:"ntt_cases"`
}

type BN254NTTCaseJSON struct {
	Name                   string     `json:"name"`
	Size                   int        `json:"size"`
	InputMontLE            []string   `json:"input_mont_le"`
	ForwardExpectedLE      []string   `json:"forward_expected_le"`
	InverseExpectedLE      []string   `json:"inverse_expected_le"`
	StageTwiddlesLE        [][]string `json:"stage_twiddles_le"`
	InverseStageTwiddlesLE [][]string `json:"inverse_stage_twiddles_le"`
	InverseScaleLE         string     `json:"inverse_scale_le"`
}

func BuildBN254NTTDomainFile(minLog, maxLog int) BN254NTTDomainFileJSON {
	out := BN254NTTDomainFileJSON{
		Domains: make([]BN254NTTDomainJSON, 0, maxLog-minLog+1),
	}
	for logN := minLog; logN <= maxLog; logN++ {
		size := 1 << logN
		domain := gnarkfft.NewDomain(uint64(size))
		out.Domains = append(out.Domains, BN254NTTDomainJSON{
			LogN:              logN,
			Size:              size,
			OmegaHex:          domain.Generator.BigInt(new(big.Int)).Text(16),
			OmegaInvHex:       domain.GeneratorInv.BigInt(new(big.Int)).Text(16),
			CardinalityInvHex: domain.CardinalityInv.BigInt(new(big.Int)).Text(16),
		})
	}
	return out
}

func BuildBN254NTTVectors() BN254NTTVectorsJSON {
	return BN254NTTVectorsJSON{
		NTTCases: []BN254NTTCaseJSON{
			buildBN254NTTCase("n8_random", 8, rand.New(rand.NewSource(2026040203))),
			buildBN254NTTCase("n16_random", 16, rand.New(rand.NewSource(2026040204))),
		},
	}
}

func buildBN254NTTCase(name string, size int, rng *rand.Rand) BN254NTTCaseJSON {
	domain := gnarkfft.NewDomain(uint64(size))
	twiddles, err := domain.Twiddles()
	if err != nil {
		panic(err)
	}
	twiddlesInv, err := domain.TwiddlesInv()
	if err != nil {
		panic(err)
	}

	input := make([]gnarkbn254fr.Element, size)
	for i := range input {
		input[i].SetBigInt(randomBN254FRFieldBigInt(rng))
	}

	forward := make([]gnarkbn254fr.Element, size)
	copy(forward, input)
	utils.BitReverse(forward)
	domain.FFT(forward, gnarkfft.DIT)

	inverse := make([]gnarkbn254fr.Element, size)
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
			stageTwiddles[stage-1][i] = bn254FrElementHex(src[i])
			inverseStageTwiddles[stage-1][i] = bn254FrElementHex(srcInv[i])
		}
	}

	return BN254NTTCaseJSON{
		Name:                   name,
		Size:                   size,
		InputMontLE:            encodeBN254FRBatch(input),
		ForwardExpectedLE:      encodeBN254FRBatch(forward),
		InverseExpectedLE:      encodeBN254FRBatch(inverse),
		StageTwiddlesLE:        stageTwiddles,
		InverseStageTwiddlesLE: inverseStageTwiddles,
		InverseScaleLE:         bn254FrElementHex(domain.CardinalityInv),
	}
}

func encodeBN254FRBatch(in []gnarkbn254fr.Element) []string {
	out := make([]string, len(in))
	for i := range in {
		out[i] = bn254FrElementHex(in[i])
	}
	return out
}
