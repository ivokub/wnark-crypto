package main

import (
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"path/filepath"

	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkfp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
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

type phase7Vectors struct {
	GeneratorAffine affinePoint         `json:"generator_affine"`
	OneMontZ        string              `json:"one_mont_z"`
	ScalarCases     []scalarMulCase     `json:"scalar_cases"`
	BaseCases       []scalarMulBaseCase `json:"base_cases"`
}

func main() {
	out := flag.String("out", "testdata/vectors/g1/bn254_phase7_scalar_mul.json", "output vector path")
	flag.Parse()

	if err := run(*out); err != nil {
		panic(err)
	}
}

func run(outPath string) error {
	_, _, genAff, _ := gnarkbn254.Generators()
	infAff := new(gnarkbn254.G1Affine).SetInfinity()
	fiveGen := scalarMulGenerator(newScalarUint64(5))
	oneTwentyThreeGen := scalarMulGenerator(newScalarUint64(123))
	modMinusOne := new(big.Int).Sub(gnarkfr.Modulus(), big.NewInt(1))

	scalarCases := []struct {
		name   string
		base   *gnarkbn254.G1Affine
		scalar gnarkfr.Element
	}{
		{name: "gen_times_zero", base: &genAff, scalar: newScalarUint64(0)},
		{name: "gen_times_one", base: &genAff, scalar: newScalarUint64(1)},
		{name: "gen_times_two", base: &genAff, scalar: newScalarUint64(2)},
		{name: "five_gen_times_seventeen", base: fiveGen, scalar: newScalarUint64(17)},
		{name: "infinity_times_one_twenty_three", base: infAff, scalar: newScalarUint64(123)},
		{name: "one_twenty_three_times_forty_two", base: oneTwentyThreeGen, scalar: newScalarUint64(42)},
		{name: "gen_times_q_minus_one", base: &genAff, scalar: newScalarBig(modMinusOne)},
	}

	baseCases := []struct {
		name   string
		scalar gnarkfr.Element
	}{
		{name: "base_zero", scalar: newScalarUint64(0)},
		{name: "base_one", scalar: newScalarUint64(1)},
		{name: "base_two", scalar: newScalarUint64(2)},
		{name: "base_one_twenty_three", scalar: newScalarUint64(123)},
		{name: "base_q_minus_one", scalar: newScalarBig(modMinusOne)},
	}

	out := phase7Vectors{
		GeneratorAffine: affineToJSON(&genAff),
		OneMontZ:        elementToHex(montOne()),
		ScalarCases:     make([]scalarMulCase, len(scalarCases)),
		BaseCases:       make([]scalarMulBaseCase, len(baseCases)),
	}

	for i, tc := range scalarCases {
		var expected gnarkbn254.G1Affine
		expected.ScalarMultiplication(tc.base, scalarToBig(tc.scalar))
		out.ScalarCases[i] = scalarMulCase{
			Name:            tc.name,
			BaseAffine:      affineToJSON(tc.base),
			ScalarBytesLE:   scalarToHex(tc.scalar),
			ScalarMulAffine: affineOutputToJSON(&expected),
		}
	}

	for i, tc := range baseCases {
		var expected gnarkbn254.G1Affine
		expected.ScalarMultiplicationBase(scalarToBig(tc.scalar))
		out.BaseCases[i] = scalarMulBaseCase{
			Name:                tc.name,
			ScalarBytesLE:       scalarToHex(tc.scalar),
			ScalarMulBaseAffine: affineOutputToJSON(&expected),
		}
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		return err
	}
	fmt.Printf("wrote %s\n", outPath)
	return nil
}

func scalarMulGenerator(s gnarkfr.Element) *gnarkbn254.G1Affine {
	_, _, genAff, _ := gnarkbn254.Generators()
	var out gnarkbn254.G1Affine
	out.ScalarMultiplication(&genAff, scalarToBig(s))
	return &out
}

func newScalarUint64(v uint64) gnarkfr.Element {
	var out gnarkfr.Element
	out.SetUint64(v)
	return out
}

func newScalarBig(v *big.Int) gnarkfr.Element {
	var out gnarkfr.Element
	out.SetBigInt(v)
	return out
}

func scalarToBig(v gnarkfr.Element) *big.Int {
	return new(big.Int).SetBytes(regularLEBytes(words4ToBytes(v[:])))
}

func scalarToHex(v gnarkfr.Element) string {
	return hex.EncodeToString(words4ToBytes(v[:]))
}

func affineToJSON(p *gnarkbn254.G1Affine) affinePoint {
	return affinePoint{
		XBytesLE: elementToHex(p.X),
		YBytesLE: elementToHex(p.Y),
	}
}

func affineOutputToJSON(p *gnarkbn254.G1Affine) jacPoint {
	if p.IsInfinity() {
		return jacPoint{
			XBytesLE: zeroHex(),
			YBytesLE: zeroHex(),
			ZBytesLE: zeroHex(),
		}
	}
	return jacPoint{
		XBytesLE: elementToHex(p.X),
		YBytesLE: elementToHex(p.Y),
		ZBytesLE: elementToHex(montOne()),
	}
}

func elementToHex(v gnarkfp.Element) string {
	return hex.EncodeToString(words4ToBytes(v[:]))
}

func words4ToBytes(words []uint64) []byte {
	out := make([]byte, 32)
	for i, word := range words {
		base := i * 8
		out[base+0] = byte(word)
		out[base+1] = byte(word >> 8)
		out[base+2] = byte(word >> 16)
		out[base+3] = byte(word >> 24)
		out[base+4] = byte(word >> 32)
		out[base+5] = byte(word >> 40)
		out[base+6] = byte(word >> 48)
		out[base+7] = byte(word >> 56)
	}
	return out
}

func regularLEBytes(in []byte) []byte {
	out := make([]byte, len(in))
	for i := range in {
		out[len(in)-1-i] = in[i]
	}
	return out
}

func montOne() gnarkfp.Element {
	var one gnarkfp.Element
	one.SetOne()
	return one
}

func zeroHex() string {
	return "0000000000000000000000000000000000000000000000000000000000000000"
}
