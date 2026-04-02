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

type msmCase struct {
	Name           string        `json:"name"`
	BasesAffine    []affinePoint `json:"bases_affine"`
	ScalarsBytesLE []string      `json:"scalars_bytes_le"`
	ExpectedAffine jacPoint      `json:"expected_affine"`
}

type phase8Vectors struct {
	TermsPerInstance int       `json:"terms_per_instance"`
	MSMCases         []msmCase `json:"msm_cases"`
	OneMontZ         string    `json:"one_mont_z"`
}

func main() {
	out := flag.String("out", "testdata/vectors/g1/bn254_phase8_msm.json", "output vector path")
	flag.Parse()

	if err := run(*out); err != nil {
		panic(err)
	}
}

func run(outPath string) error {
	_, _, genAff, _ := gnarkbn254.Generators()
	infAff := new(gnarkbn254.G1Affine).SetInfinity()
	five := scalarMulGenerator(5)
	seventeen := scalarMulGenerator(17)
	oneTwentyThree := scalarMulGenerator(123)
	twoHundredEleven := scalarMulGenerator(211)
	modMinusOne := new(big.Int).Sub(gnarkfr.Modulus(), big.NewInt(1))

	cases := []struct {
		name    string
		bases   []*gnarkbn254.G1Affine
		scalars []gnarkfr.Element
	}{
		{
			name:    "single_generator",
			bases:   []*gnarkbn254.G1Affine{&genAff, infAff, infAff, infAff},
			scalars: []gnarkfr.Element{scalarUint64(1), scalarUint64(0), scalarUint64(0), scalarUint64(0)},
		},
		{
			name:    "simple_linear_combo",
			bases:   []*gnarkbn254.G1Affine{&genAff, five, seventeen, oneTwentyThree},
			scalars: []gnarkfr.Element{scalarUint64(3), scalarUint64(4), scalarUint64(5), scalarUint64(6)},
		},
		{
			name:    "includes_infinity_and_zero_scalar",
			bases:   []*gnarkbn254.G1Affine{infAff, five, infAff, seventeen},
			scalars: []gnarkfr.Element{scalarUint64(19), scalarUint64(0), scalarUint64(7), scalarUint64(9)},
		},
		{
			name:    "q_minus_one_mix",
			bases:   []*gnarkbn254.G1Affine{&genAff, five, twoHundredEleven, oneTwentyThree},
			scalars: []gnarkfr.Element{scalarBig(modMinusOne), scalarUint64(2), scalarUint64(0), scalarUint64(13)},
		},
	}

	out := phase8Vectors{
		TermsPerInstance: 4,
		MSMCases:         make([]msmCase, len(cases)),
		OneMontZ:         elementToHex(montOne()),
	}

	for i, tc := range cases {
		expected := naiveMSM(tc.bases, tc.scalars)
		out.MSMCases[i] = msmCase{
			Name:           tc.name,
			BasesAffine:    encodeAffineBatch(tc.bases),
			ScalarsBytesLE: encodeScalarBatch(tc.scalars),
			ExpectedAffine: affineOutputToJSON(expected),
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

func scalarMulGenerator(v uint64) *gnarkbn254.G1Affine {
	_, _, genAff, _ := gnarkbn254.Generators()
	var out gnarkbn254.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func scalarUint64(v uint64) gnarkfr.Element {
	var out gnarkfr.Element
	out.SetUint64(v)
	return out
}

func scalarBig(v *big.Int) gnarkfr.Element {
	var out gnarkfr.Element
	out.SetBigInt(v)
	return out
}

func encodeAffineBatch(in []*gnarkbn254.G1Affine) []affinePoint {
	out := make([]affinePoint, len(in))
	for i, point := range in {
		out[i] = affineToJSON(point)
	}
	return out
}

func encodeScalarBatch(in []gnarkfr.Element) []string {
	out := make([]string, len(in))
	for i, scalar := range in {
		out[i] = scalarToHex(scalar)
	}
	return out
}

func scalarToHex(v gnarkfr.Element) string {
	bytesBE := v.Bytes()
	return hex.EncodeToString(regularToLittleEndian(bytesBE[:]))
}

func scalarToBig(v gnarkfr.Element) *big.Int {
	var out big.Int
	return v.BigInt(&out)
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

func regularToLittleEndian(in []byte) []byte {
	out := make([]byte, len(in))
	for i := range in {
		out[len(in)-1-i] = in[i]
	}
	return out
}

func naiveMSM(bases []*gnarkbn254.G1Affine, scalars []gnarkfr.Element) *gnarkbn254.G1Affine {
	sum := new(gnarkbn254.G1Affine).SetInfinity()
	for i := range bases {
		var term gnarkbn254.G1Affine
		term.ScalarMultiplication(bases[i], scalarToBig(scalars[i]))
		sum.Add(sum, &term)
	}
	return sum
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
	var out [32]byte
	for i, word := range [4]uint64(v) {
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
	return hex.EncodeToString(out[:])
}

func montOne() gnarkfp.Element {
	var one gnarkfp.Element
	one.SetOne()
	return one
}

func zeroHex() string {
	return "0000000000000000000000000000000000000000000000000000000000000000"
}
