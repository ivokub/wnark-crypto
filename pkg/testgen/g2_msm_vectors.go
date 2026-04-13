package testgen

import (
	"math/big"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkbls12381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkbn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type G2MSMCaseJSON struct {
	Name           string              `json:"name"`
	BasesAffine    []G2AffinePointJSON `json:"bases_affine"`
	ScalarsBytesLE []string            `json:"scalars_bytes_le"`
	ExpectedAffine G2JacobianPointJSON `json:"expected_affine"`
}

type G2MSMVectorsJSON struct {
	TermsPerInstance int             `json:"terms_per_instance"`
	MSMCases         []G2MSMCaseJSON `json:"msm_cases"`
}

func BuildBN254G2MSMVectors() G2MSMVectorsJSON {
	_, _, _, genAff := gnarkbn254.Generators()
	infAff := new(gnarkbn254.G2Affine).SetInfinity()
	five := scalarMulBN254G2Ops(5)
	seventeen := scalarMulBN254G2Ops(17)
	oneTwentyThree := scalarMulBN254G2Ops(123)
	twoHundredEleven := scalarMulBN254G2Ops(211)
	modMinusOne := new(big.Int).Sub(gnarkbn254fr.Modulus(), big.NewInt(1))

	cases := []struct {
		name    string
		bases   []*gnarkbn254.G2Affine
		scalars []gnarkbn254fr.Element
	}{
		{
			name:    "single_generator",
			bases:   []*gnarkbn254.G2Affine{&genAff, infAff, infAff, infAff},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarUint64(1), bn254MSMScalarUint64(0), bn254MSMScalarUint64(0), bn254MSMScalarUint64(0)},
		},
		{
			name:    "simple_linear_combo",
			bases:   []*gnarkbn254.G2Affine{&genAff, five, seventeen, oneTwentyThree},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarUint64(3), bn254MSMScalarUint64(4), bn254MSMScalarUint64(5), bn254MSMScalarUint64(6)},
		},
		{
			name:    "includes_infinity_and_zero_scalar",
			bases:   []*gnarkbn254.G2Affine{infAff, five, infAff, seventeen},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarUint64(19), bn254MSMScalarUint64(0), bn254MSMScalarUint64(7), bn254MSMScalarUint64(9)},
		},
		{
			name:    "q_minus_one_mix",
			bases:   []*gnarkbn254.G2Affine{&genAff, five, twoHundredEleven, oneTwentyThree},
			scalars: []gnarkbn254fr.Element{bn254MSMScalarBig(modMinusOne), bn254MSMScalarUint64(2), bn254MSMScalarUint64(0), bn254MSMScalarUint64(13)},
		},
	}

	out := G2MSMVectorsJSON{
		TermsPerInstance: 4,
		MSMCases:         make([]G2MSMCaseJSON, len(cases)),
	}

	for i, tc := range cases {
		expected := naiveBN254G2MSM(tc.bases, tc.scalars)
		out.MSMCases[i] = G2MSMCaseJSON{
			Name:           tc.name,
			BasesAffine:    encodeBN254G2AffineBatch(tc.bases),
			ScalarsBytesLE: encodeBN254ScalarBatch(tc.scalars),
			ExpectedAffine: bn254G2AffineOutputToJSON(expected),
		}
	}
	return out
}

func BuildBLS12381G2MSMVectors() G2MSMVectorsJSON {
	_, _, _, genAff := gnarkbls12381.Generators()
	infAff := new(gnarkbls12381.G2Affine).SetInfinity()
	five := scalarMulBLS12381G2Ops(5)
	seventeen := scalarMulBLS12381G2Ops(17)
	oneTwentyThree := scalarMulBLS12381G2Ops(123)
	twoHundredEleven := scalarMulBLS12381G2Ops(211)
	modMinusOne := new(big.Int).Sub(gnarkbls12381fr.Modulus(), big.NewInt(1))

	cases := []struct {
		name    string
		bases   []*gnarkbls12381.G2Affine
		scalars []gnarkbls12381fr.Element
	}{
		{
			name:    "single_generator",
			bases:   []*gnarkbls12381.G2Affine{&genAff, infAff, infAff, infAff},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarUint64(1), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(0)},
		},
		{
			name:    "simple_linear_combo",
			bases:   []*gnarkbls12381.G2Affine{&genAff, five, seventeen, oneTwentyThree},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarUint64(3), bls12381MSMScalarUint64(4), bls12381MSMScalarUint64(5), bls12381MSMScalarUint64(6)},
		},
		{
			name:    "includes_infinity_and_zero_scalar",
			bases:   []*gnarkbls12381.G2Affine{infAff, five, infAff, seventeen},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarUint64(19), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(7), bls12381MSMScalarUint64(9)},
		},
		{
			name:    "q_minus_one_mix",
			bases:   []*gnarkbls12381.G2Affine{&genAff, five, twoHundredEleven, oneTwentyThree},
			scalars: []gnarkbls12381fr.Element{bls12381MSMScalarBig(modMinusOne), bls12381MSMScalarUint64(2), bls12381MSMScalarUint64(0), bls12381MSMScalarUint64(13)},
		},
	}

	out := G2MSMVectorsJSON{
		TermsPerInstance: 4,
		MSMCases:         make([]G2MSMCaseJSON, len(cases)),
	}

	for i, tc := range cases {
		expected := naiveBLS12381G2MSM(tc.bases, tc.scalars)
		out.MSMCases[i] = G2MSMCaseJSON{
			Name:           tc.name,
			BasesAffine:    encodeBLS12381G2AffineBatch(tc.bases),
			ScalarsBytesLE: encodeBLS12381ScalarBatch(tc.scalars),
			ExpectedAffine: bls12381G2AffineOutputToJSON(expected),
		}
	}
	return out
}

func encodeBN254G2AffineBatch(in []*gnarkbn254.G2Affine) []G2AffinePointJSON {
	out := make([]G2AffinePointJSON, len(in))
	for i, point := range in {
		out[i] = bn254G2AffineToJSON(point)
	}
	return out
}

func encodeBLS12381G2AffineBatch(in []*gnarkbls12381.G2Affine) []G2AffinePointJSON {
	out := make([]G2AffinePointJSON, len(in))
	for i, point := range in {
		out[i] = bls12381G2AffineToJSON(point)
	}
	return out
}

func naiveBN254G2MSM(bases []*gnarkbn254.G2Affine, scalars []gnarkbn254fr.Element) *gnarkbn254.G2Affine {
	sum := new(gnarkbn254.G2Affine).SetInfinity()
	for i := range bases {
		var term gnarkbn254.G2Affine
		term.ScalarMultiplication(bases[i], bn254ScalarToBig(scalars[i]))
		sum.Add(sum, &term)
	}
	return sum
}

func naiveBLS12381G2MSM(bases []*gnarkbls12381.G2Affine, scalars []gnarkbls12381fr.Element) *gnarkbls12381.G2Affine {
	sum := new(gnarkbls12381.G2Affine).SetInfinity()
	for i := range bases {
		var term gnarkbls12381.G2Affine
		term.ScalarMultiplication(bases[i], bls12381ScalarToBig(scalars[i]))
		sum.Add(sum, &term)
	}
	return sum
}
