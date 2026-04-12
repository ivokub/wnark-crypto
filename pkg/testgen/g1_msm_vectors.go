package testgen

import (
	"math/big"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkbls12381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkbn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type G1MSMCaseJSON struct {
	Name           string            `json:"name"`
	BasesAffine    []AffinePointJSON `json:"bases_affine"`
	ScalarsBytesLE []string          `json:"scalars_bytes_le"`
	ExpectedAffine JacobianPointJSON `json:"expected_affine"`
}

type G1MSMVectorsJSON struct {
	TermsPerInstance int             `json:"terms_per_instance"`
	MSMCases         []G1MSMCaseJSON `json:"msm_cases"`
	OneMontZ         string          `json:"one_mont_z"`
}

func BuildBN254G1MSMVectors() G1MSMVectorsJSON {
	_, _, genAff, _ := gnarkbn254.Generators()
	infAff := new(gnarkbn254.G1Affine).SetInfinity()
	five := scalarMulBN254G1MSM(5)
	seventeen := scalarMulBN254G1MSM(17)
	oneTwentyThree := scalarMulBN254G1MSM(123)
	twoHundredEleven := scalarMulBN254G1MSM(211)
	modMinusOne := new(big.Int).Sub(gnarkbn254fr.Modulus(), big.NewInt(1))

	cases := []struct {
		name    string
		bases   []*gnarkbn254.G1Affine
		scalars []gnarkbn254fr.Element
	}{
		{
			name:    "single_generator",
			bases:   []*gnarkbn254.G1Affine{&genAff, infAff, infAff, infAff},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarUint64(1), bn254MSMScalarUint64(0), bn254MSMScalarUint64(0), bn254MSMScalarUint64(0)},
		},
		{
			name:    "simple_linear_combo",
			bases:   []*gnarkbn254.G1Affine{&genAff, five, seventeen, oneTwentyThree},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarUint64(3), bn254MSMScalarUint64(4), bn254MSMScalarUint64(5), bn254MSMScalarUint64(6)},
		},
		{
			name:    "includes_infinity_and_zero_scalar",
			bases:   []*gnarkbn254.G1Affine{infAff, five, infAff, seventeen},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarUint64(19), bn254MSMScalarUint64(0), bn254MSMScalarUint64(7), bn254MSMScalarUint64(9)},
		},
		{
			name:    "q_minus_one_mix",
			bases:   []*gnarkbn254.G1Affine{&genAff, five, twoHundredEleven, oneTwentyThree},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarBig(modMinusOne), bn254MSMScalarUint64(2), bn254MSMScalarUint64(0), bn254MSMScalarUint64(13)},
		},
	}

	out := G1MSMVectorsJSON{
		TermsPerInstance: 4,
		MSMCases:         make([]G1MSMCaseJSON, len(cases)),
		OneMontZ:         bn254ElementToHex(montOneBN254()),
	}

	for i, tc := range cases {
		expected := naiveBN254MSM(tc.bases, tc.scalars)
		out.MSMCases[i] = G1MSMCaseJSON{
			Name:           tc.name,
			BasesAffine:    encodeBN254AffineBatch(tc.bases),
			ScalarsBytesLE: encodeBN254ScalarBatch(tc.scalars),
			ExpectedAffine: bn254AffineOutputToJSON(expected),
		}
	}

	return out
}

func BuildBLS12381G1MSMVectors() G1MSMVectorsJSON {
	_, _, genAff, _ := gnarkbls12381.Generators()
	infAff := new(gnarkbls12381.G1Affine).SetInfinity()
	five := scalarMulBLS12381G1MSM(5)
	seventeen := scalarMulBLS12381G1MSM(17)
	oneTwentyThree := scalarMulBLS12381G1MSM(123)
	twoHundredEleven := scalarMulBLS12381G1MSM(211)
	modMinusOne := new(big.Int).Sub(gnarkbls12381fr.Modulus(), big.NewInt(1))

	cases := []struct {
		name    string
		bases   []*gnarkbls12381.G1Affine
		scalars []gnarkbls12381fr.Element
	}{
		{
			name:    "single_generator",
			bases:   []*gnarkbls12381.G1Affine{&genAff, infAff, infAff, infAff},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarUint64(1), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(0)},
		},
		{
			name:    "simple_linear_combo",
			bases:   []*gnarkbls12381.G1Affine{&genAff, five, seventeen, oneTwentyThree},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarUint64(3), bls12381MSMScalarUint64(4), bls12381MSMScalarUint64(5), bls12381MSMScalarUint64(6)},
		},
		{
			name:    "includes_infinity_and_zero_scalar",
			bases:   []*gnarkbls12381.G1Affine{infAff, five, infAff, seventeen},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarUint64(19), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(7), bls12381MSMScalarUint64(9)},
		},
		{
			name:    "q_minus_one_mix",
			bases:   []*gnarkbls12381.G1Affine{&genAff, five, twoHundredEleven, oneTwentyThree},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarBig(modMinusOne), bls12381MSMScalarUint64(2), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(13)},
		},
	}

	out := G1MSMVectorsJSON{
		TermsPerInstance: 4,
		MSMCases:         make([]G1MSMCaseJSON, len(cases)),
		OneMontZ:         bls12381ElementToHex(montOneBLS12381()),
	}

	for i, tc := range cases {
		expected := naiveBLS12381MSM(tc.bases, tc.scalars)
		out.MSMCases[i] = G1MSMCaseJSON{
			Name:           tc.name,
			BasesAffine:    encodeBLS12381AffineBatch(tc.bases),
			ScalarsBytesLE: encodeBLS12381ScalarBatch(tc.scalars),
			ExpectedAffine: bls12381AffineOutputToJSON(expected),
		}
	}

	return out
}

func scalarMulBN254G1MSM(v uint64) *gnarkbn254.G1Affine {
	_, _, genAff, _ := gnarkbn254.Generators()
	var out gnarkbn254.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func scalarMulBLS12381G1MSM(v uint64) *gnarkbls12381.G1Affine {
	_, _, genAff, _ := gnarkbls12381.Generators()
	var out gnarkbls12381.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func bn254MSMScalarUint64(v uint64) gnarkbn254fr.Element {
	var out gnarkbn254fr.Element
	out.SetUint64(v)
	return out
}

func bn254MSMScalarBig(v *big.Int) gnarkbn254fr.Element {
	var out gnarkbn254fr.Element
	out.SetBigInt(v)
	return out
}

func bls12381MSMScalarUint64(v uint64) gnarkbls12381fr.Element {
	var out gnarkbls12381fr.Element
	out.SetUint64(v)
	return out
}

func bls12381MSMScalarBig(v *big.Int) gnarkbls12381fr.Element {
	var out gnarkbls12381fr.Element
	out.SetBigInt(v)
	return out
}

func encodeBN254AffineBatch(in []*gnarkbn254.G1Affine) []AffinePointJSON {
	out := make([]AffinePointJSON, len(in))
	for i, point := range in {
		out[i] = bn254AffineToJSON(point)
	}
	return out
}

func encodeBLS12381AffineBatch(in []*gnarkbls12381.G1Affine) []AffinePointJSON {
	out := make([]AffinePointJSON, len(in))
	for i, point := range in {
		out[i] = bls12381AffineToJSON(point)
	}
	return out
}

func encodeBN254ScalarBatch(in []gnarkbn254fr.Element) []string {
	out := make([]string, len(in))
	for i, scalar := range in {
		out[i] = bn254ScalarToHex(scalar)
	}
	return out
}

func encodeBLS12381ScalarBatch(in []gnarkbls12381fr.Element) []string {
	out := make([]string, len(in))
	for i, scalar := range in {
		out[i] = bls12381ScalarToHex(scalar)
	}
	return out
}

func naiveBN254MSM(bases []*gnarkbn254.G1Affine, scalars []gnarkbn254fr.Element) *gnarkbn254.G1Affine {
	sum := new(gnarkbn254.G1Affine).SetInfinity()
	for i := range bases {
		var term gnarkbn254.G1Affine
		term.ScalarMultiplication(bases[i], bn254ScalarToBig(scalars[i]))
		sum.Add(sum, &term)
	}
	return sum
}

func naiveBLS12381MSM(bases []*gnarkbls12381.G1Affine, scalars []gnarkbls12381fr.Element) *gnarkbls12381.G1Affine {
	sum := new(gnarkbls12381.G1Affine).SetInfinity()
	for i := range bases {
		var term gnarkbls12381.G1Affine
		term.ScalarMultiplication(bases[i], bls12381ScalarToBig(scalars[i]))
		sum.Add(sum, &term)
	}
	return sum
}
