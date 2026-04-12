package testgen

import (
	"encoding/hex"
	"math/big"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkbls12381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	gnarkbls12381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkbn254fp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	gnarkbn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type AffinePointJSON struct {
	XBytesLE string `json:"x_bytes_le"`
	YBytesLE string `json:"y_bytes_le"`
}

type JacobianPointJSON struct {
	XBytesLE string `json:"x_bytes_le"`
	YBytesLE string `json:"y_bytes_le"`
	ZBytesLE string `json:"z_bytes_le"`
}

type G1ScalarMulCaseJSON struct {
	Name            string          `json:"name"`
	BaseAffine      AffinePointJSON `json:"base_affine"`
	ScalarBytesLE   string          `json:"scalar_bytes_le"`
	ScalarMulAffine JacobianPointJSON `json:"scalar_mul_affine"`
}

type G1ScalarMulBaseCaseJSON struct {
	Name                string            `json:"name"`
	ScalarBytesLE       string            `json:"scalar_bytes_le"`
	ScalarMulBaseAffine JacobianPointJSON `json:"scalar_mul_base_affine"`
}

type G1ScalarMulVectorsJSON struct {
	GeneratorAffine AffinePointJSON         `json:"generator_affine"`
	OneMontZ        string                  `json:"one_mont_z"`
	ScalarCases     []G1ScalarMulCaseJSON   `json:"scalar_cases"`
	BaseCases       []G1ScalarMulBaseCaseJSON `json:"base_cases"`
}

func BuildBN254G1ScalarVectors() G1ScalarMulVectorsJSON {
	_, _, genAff, _ := gnarkbn254.Generators()
	infAff := new(gnarkbn254.G1Affine).SetInfinity()
	fiveGen := scalarMulBN254Generator(newBN254ScalarUint64(5))
	oneTwentyThreeGen := scalarMulBN254Generator(newBN254ScalarUint64(123))
	modMinusOne := new(big.Int).Sub(gnarkbn254fr.Modulus(), big.NewInt(1))

	scalarCases := []struct {
		name   string
		base   *gnarkbn254.G1Affine
		scalar gnarkbn254fr.Element
	}{
		{name: "gen_times_zero", base: &genAff, scalar: newBN254ScalarUint64(0)},
		{name: "gen_times_one", base: &genAff, scalar: newBN254ScalarUint64(1)},
		{name: "gen_times_two", base: &genAff, scalar: newBN254ScalarUint64(2)},
		{name: "five_gen_times_seventeen", base: fiveGen, scalar: newBN254ScalarUint64(17)},
		{name: "infinity_times_one_twenty_three", base: infAff, scalar: newBN254ScalarUint64(123)},
		{name: "one_twenty_three_times_forty_two", base: oneTwentyThreeGen, scalar: newBN254ScalarUint64(42)},
		{name: "gen_times_q_minus_one", base: &genAff, scalar: newBN254ScalarBig(modMinusOne)},
	}

	baseCases := []struct {
		name   string
		scalar gnarkbn254fr.Element
	}{
		{name: "base_zero", scalar: newBN254ScalarUint64(0)},
		{name: "base_one", scalar: newBN254ScalarUint64(1)},
		{name: "base_two", scalar: newBN254ScalarUint64(2)},
		{name: "base_one_twenty_three", scalar: newBN254ScalarUint64(123)},
		{name: "base_q_minus_one", scalar: newBN254ScalarBig(modMinusOne)},
	}

	out := G1ScalarMulVectorsJSON{
		GeneratorAffine: bn254AffineToJSON(&genAff),
		OneMontZ:        bn254ElementToHex(montOneBN254ScalarVectors()),
		ScalarCases:     make([]G1ScalarMulCaseJSON, len(scalarCases)),
		BaseCases:       make([]G1ScalarMulBaseCaseJSON, len(baseCases)),
	}

	for i, tc := range scalarCases {
		var expected gnarkbn254.G1Affine
		expected.ScalarMultiplication(tc.base, bn254ScalarToBig(tc.scalar))
		out.ScalarCases[i] = G1ScalarMulCaseJSON{
			Name:            tc.name,
			BaseAffine:      bn254AffineToJSON(tc.base),
			ScalarBytesLE:   bn254ScalarToHex(tc.scalar),
			ScalarMulAffine: bn254AffineOutputToJSON(&expected),
		}
	}

	for i, tc := range baseCases {
		var expected gnarkbn254.G1Affine
		expected.ScalarMultiplicationBase(bn254ScalarToBig(tc.scalar))
		out.BaseCases[i] = G1ScalarMulBaseCaseJSON{
			Name:                tc.name,
			ScalarBytesLE:       bn254ScalarToHex(tc.scalar),
			ScalarMulBaseAffine: bn254AffineOutputToJSON(&expected),
		}
	}

	return out
}

func BuildBLS12381G1ScalarVectors() G1ScalarMulVectorsJSON {
	_, _, genAff, _ := gnarkbls12381.Generators()
	infAff := new(gnarkbls12381.G1Affine).SetInfinity()
	fiveGen := scalarMulBLS12381Generator(newBLS12381ScalarUint64(5))
	oneTwentyThreeGen := scalarMulBLS12381Generator(newBLS12381ScalarUint64(123))
	modMinusOne := new(big.Int).Sub(gnarkbls12381fr.Modulus(), big.NewInt(1))

	scalarCases := []struct {
		name   string
		base   *gnarkbls12381.G1Affine
		scalar gnarkbls12381fr.Element
	}{
		{name: "gen_times_zero", base: &genAff, scalar: newBLS12381ScalarUint64(0)},
		{name: "gen_times_one", base: &genAff, scalar: newBLS12381ScalarUint64(1)},
		{name: "gen_times_two", base: &genAff, scalar: newBLS12381ScalarUint64(2)},
		{name: "five_gen_times_seventeen", base: fiveGen, scalar: newBLS12381ScalarUint64(17)},
		{name: "infinity_times_one_twenty_three", base: infAff, scalar: newBLS12381ScalarUint64(123)},
		{name: "one_twenty_three_times_forty_two", base: oneTwentyThreeGen, scalar: newBLS12381ScalarUint64(42)},
		{name: "gen_times_q_minus_one", base: &genAff, scalar: newBLS12381ScalarBig(modMinusOne)},
	}

	baseCases := []struct {
		name   string
		scalar gnarkbls12381fr.Element
	}{
		{name: "base_zero", scalar: newBLS12381ScalarUint64(0)},
		{name: "base_one", scalar: newBLS12381ScalarUint64(1)},
		{name: "base_two", scalar: newBLS12381ScalarUint64(2)},
		{name: "base_one_twenty_three", scalar: newBLS12381ScalarUint64(123)},
		{name: "base_q_minus_one", scalar: newBLS12381ScalarBig(modMinusOne)},
	}

	out := G1ScalarMulVectorsJSON{
		GeneratorAffine: bls12381AffineToJSON(&genAff),
		OneMontZ:        bls12381ElementToHex(montOneBLS12381ScalarVectors()),
		ScalarCases:     make([]G1ScalarMulCaseJSON, len(scalarCases)),
		BaseCases:       make([]G1ScalarMulBaseCaseJSON, len(baseCases)),
	}

	for i, tc := range scalarCases {
		var expected gnarkbls12381.G1Affine
		expected.ScalarMultiplication(tc.base, bls12381ScalarToBig(tc.scalar))
		out.ScalarCases[i] = G1ScalarMulCaseJSON{
			Name:            tc.name,
			BaseAffine:      bls12381AffineToJSON(tc.base),
			ScalarBytesLE:   bls12381ScalarToHex(tc.scalar),
			ScalarMulAffine: bls12381AffineOutputToJSON(&expected),
		}
	}

	for i, tc := range baseCases {
		var expected gnarkbls12381.G1Affine
		expected.ScalarMultiplicationBase(bls12381ScalarToBig(tc.scalar))
		out.BaseCases[i] = G1ScalarMulBaseCaseJSON{
			Name:                tc.name,
			ScalarBytesLE:       bls12381ScalarToHex(tc.scalar),
			ScalarMulBaseAffine: bls12381AffineOutputToJSON(&expected),
		}
	}

	return out
}

func scalarMulBN254Generator(s gnarkbn254fr.Element) *gnarkbn254.G1Affine {
	_, _, genAff, _ := gnarkbn254.Generators()
	var out gnarkbn254.G1Affine
	out.ScalarMultiplication(&genAff, bn254ScalarToBig(s))
	return &out
}

func scalarMulBLS12381Generator(s gnarkbls12381fr.Element) *gnarkbls12381.G1Affine {
	_, _, genAff, _ := gnarkbls12381.Generators()
	var out gnarkbls12381.G1Affine
	out.ScalarMultiplication(&genAff, bls12381ScalarToBig(s))
	return &out
}

func newBN254ScalarUint64(v uint64) gnarkbn254fr.Element {
	var out gnarkbn254fr.Element
	out.SetUint64(v)
	return out
}

func newBLS12381ScalarUint64(v uint64) gnarkbls12381fr.Element {
	var out gnarkbls12381fr.Element
	out.SetUint64(v)
	return out
}

func newBN254ScalarBig(v *big.Int) gnarkbn254fr.Element {
	var out gnarkbn254fr.Element
	out.SetBigInt(v)
	return out
}

func newBLS12381ScalarBig(v *big.Int) gnarkbls12381fr.Element {
	var out gnarkbls12381fr.Element
	out.SetBigInt(v)
	return out
}

func bn254ScalarToBig(v gnarkbn254fr.Element) *big.Int {
	var out big.Int
	return v.BigInt(&out)
}

func bls12381ScalarToBig(v gnarkbls12381fr.Element) *big.Int {
	var out big.Int
	return v.BigInt(&out)
}

func bn254ScalarToHex(v gnarkbn254fr.Element) string {
	bytesBE := v.Bytes()
	return hex.EncodeToString(regularToLittleEndian(bytesBE[:]))
}

func bls12381ScalarToHex(v gnarkbls12381fr.Element) string {
	bytesBE := v.Bytes()
	return hex.EncodeToString(regularToLittleEndian(bytesBE[:]))
}

func bn254AffineToJSON(p *gnarkbn254.G1Affine) AffinePointJSON {
	return AffinePointJSON{
		XBytesLE: bn254ElementToHex(p.X),
		YBytesLE: bn254ElementToHex(p.Y),
	}
}

func bls12381AffineToJSON(p *gnarkbls12381.G1Affine) AffinePointJSON {
	return AffinePointJSON{
		XBytesLE: bls12381ElementToHex(p.X),
		YBytesLE: bls12381ElementToHex(p.Y),
	}
}

func bn254AffineOutputToJSON(p *gnarkbn254.G1Affine) JacobianPointJSON {
	if p.IsInfinity() {
		return JacobianPointJSON{
			XBytesLE: zeroHexBN254(),
			YBytesLE: zeroHexBN254(),
			ZBytesLE: zeroHexBN254(),
		}
	}
	return JacobianPointJSON{
		XBytesLE: bn254ElementToHex(p.X),
		YBytesLE: bn254ElementToHex(p.Y),
		ZBytesLE: bn254ElementToHex(montOneBN254ScalarVectors()),
	}
}

func bls12381AffineOutputToJSON(p *gnarkbls12381.G1Affine) JacobianPointJSON {
	if p.IsInfinity() {
		return JacobianPointJSON{
			XBytesLE: zeroHexBLS12381(),
			YBytesLE: zeroHexBLS12381(),
			ZBytesLE: zeroHexBLS12381(),
		}
	}
	return JacobianPointJSON{
		XBytesLE: bls12381ElementToHex(p.X),
		YBytesLE: bls12381ElementToHex(p.Y),
		ZBytesLE: bls12381ElementToHex(montOneBLS12381ScalarVectors()),
	}
}

func bn254ElementToHex(v gnarkbn254fp.Element) string {
	return hex.EncodeToString(words4ToBytes(v[:]))
}

func bls12381ElementToHex(v gnarkbls12381fp.Element) string {
	return hex.EncodeToString(words6ToBytes(v[:]))
}

func montOneBN254ScalarVectors() gnarkbn254fp.Element {
	var one gnarkbn254fp.Element
	one.SetOne()
	return one
}

func montOneBLS12381ScalarVectors() gnarkbls12381fp.Element {
	var one gnarkbls12381fp.Element
	one.SetOne()
	return one
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

func words6ToBytes(words []uint64) []byte {
	out := make([]byte, 48)
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

func zeroHexBN254() string {
	return "0000000000000000000000000000000000000000000000000000000000000000"
}

func zeroHexBLS12381() string {
	return "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
}
