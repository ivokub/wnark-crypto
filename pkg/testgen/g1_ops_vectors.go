package testgen

import (
	"math/big"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
)

type G1OpsCaseJSON struct {
	Name                string          `json:"name"`
	PAffine             AffinePointJSON `json:"p_affine"`
	QAffine             AffinePointJSON `json:"q_affine"`
	PJacobian           JacobianPointJSON `json:"p_jacobian"`
	PAffineOutput       JacobianPointJSON `json:"p_affine_output"`
	NegPJacobian        JacobianPointJSON `json:"neg_p_jacobian"`
	DoublePJacobian     JacobianPointJSON `json:"double_p_jacobian"`
	AddMixedPPlusQJacob JacobianPointJSON `json:"add_mixed_p_plus_q_jacobian"`
	AffineAddPPlusQ     JacobianPointJSON `json:"affine_add_p_plus_q"`
}

type G1OpsVectorsJSON struct {
	PointCases []G1OpsCaseJSON `json:"point_cases"`
}

func BuildBN254G1OpsVectors() G1OpsVectorsJSON {
	_, _, genAff, _ := gnarkbn254.Generators()
	infAff := new(gnarkbn254.G1Affine).SetInfinity()
	negGen := new(gnarkbn254.G1Affine).Neg(&genAff)
	five := scalarMulBN254G1Ops(5)
	seventeen := scalarMulBN254G1Ops(17)
	oneTwentyThree := scalarMulBN254G1Ops(123)

	cases := []struct {
		name string
		p    *gnarkbn254.G1Affine
		q    *gnarkbn254.G1Affine
	}{
		{name: "inf_plus_gen", p: infAff, q: &genAff},
		{name: "gen_plus_inf", p: &genAff, q: infAff},
		{name: "gen_plus_neg_gen", p: &genAff, q: negGen},
		{name: "gen_plus_gen", p: &genAff, q: &genAff},
		{name: "five_plus_seventeen", p: five, q: seventeen},
		{name: "one_twenty_three_self", p: oneTwentyThree, q: oneTwentyThree},
	}

	out := G1OpsVectorsJSON{PointCases: make([]G1OpsCaseJSON, len(cases))}
	for i, tc := range cases {
		var pJac gnarkbn254.G1Jac
		pJac.FromAffine(tc.p)

		var negP gnarkbn254.G1Jac
		negP.Neg(&pJac)

		var doubleP gnarkbn254.G1Jac
		doubleP.Double(&pJac)

		var addMixed gnarkbn254.G1Jac
		addMixed.Set(&pJac).AddMixed(tc.q)

		var pAffineOut gnarkbn254.G1Affine
		pAffineOut.FromJacobian(&pJac)

		var affineAdd gnarkbn254.G1Affine
		affineAdd.Add(tc.p, tc.q)

		out.PointCases[i] = G1OpsCaseJSON{
			Name:                tc.name,
			PAffine:             bn254AffineToJSON(tc.p),
			QAffine:             bn254AffineToJSON(tc.q),
			PJacobian:           bn254JacToJSON(&pJac),
			PAffineOutput:       bn254AffineOutputToJSON(&pAffineOut),
			NegPJacobian:        bn254JacToJSON(&negP),
			DoublePJacobian:     bn254JacToJSON(&doubleP),
			AddMixedPPlusQJacob: bn254JacToJSON(&addMixed),
			AffineAddPPlusQ:     bn254AffineOutputToJSON(&affineAdd),
		}
	}
	return out
}

func BuildBLS12381G1OpsVectors() G1OpsVectorsJSON {
	_, _, genAff, _ := gnarkbls12381.Generators()
	infAff := new(gnarkbls12381.G1Affine).SetInfinity()
	negGen := new(gnarkbls12381.G1Affine).Neg(&genAff)
	five := scalarMulBLS12381G1Ops(5)
	seventeen := scalarMulBLS12381G1Ops(17)
	oneTwentyThree := scalarMulBLS12381G1Ops(123)

	cases := []struct {
		name string
		p    *gnarkbls12381.G1Affine
		q    *gnarkbls12381.G1Affine
	}{
		{name: "inf_plus_gen", p: infAff, q: &genAff},
		{name: "gen_plus_inf", p: &genAff, q: infAff},
		{name: "gen_plus_neg_gen", p: &genAff, q: negGen},
		{name: "gen_plus_gen", p: &genAff, q: &genAff},
		{name: "five_plus_seventeen", p: five, q: seventeen},
		{name: "one_twenty_three_self", p: oneTwentyThree, q: oneTwentyThree},
	}

	out := G1OpsVectorsJSON{PointCases: make([]G1OpsCaseJSON, len(cases))}
	for i, tc := range cases {
		var pJac gnarkbls12381.G1Jac
		pJac.FromAffine(tc.p)

		var negP gnarkbls12381.G1Jac
		negP.Neg(&pJac)

		var doubleP gnarkbls12381.G1Jac
		doubleP.Double(&pJac)

		var addMixed gnarkbls12381.G1Jac
		addMixed.Set(&pJac).AddMixed(tc.q)

		var pAffineOut gnarkbls12381.G1Affine
		pAffineOut.FromJacobian(&pJac)

		var affineAdd gnarkbls12381.G1Affine
		affineAdd.Add(tc.p, tc.q)

		out.PointCases[i] = G1OpsCaseJSON{
			Name:                tc.name,
			PAffine:             bls12381AffineToJSON(tc.p),
			QAffine:             bls12381AffineToJSON(tc.q),
			PJacobian:           bls12381JacToJSON(&pJac),
			PAffineOutput:       bls12381AffineOutputToJSON(&pAffineOut),
			NegPJacobian:        bls12381JacToJSON(&negP),
			DoublePJacobian:     bls12381JacToJSON(&doubleP),
			AddMixedPPlusQJacob: bls12381JacToJSON(&addMixed),
			AffineAddPPlusQ:     bls12381AffineOutputToJSON(&affineAdd),
		}
	}
	return out
}

func scalarMulBN254G1Ops(v uint64) *gnarkbn254.G1Affine {
	_, _, genAff, _ := gnarkbn254.Generators()
	var out gnarkbn254.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func scalarMulBLS12381G1Ops(v uint64) *gnarkbls12381.G1Affine {
	_, _, genAff, _ := gnarkbls12381.Generators()
	var out gnarkbls12381.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func bn254JacToJSON(p *gnarkbn254.G1Jac) JacobianPointJSON {
	return JacobianPointJSON{
		XBytesLE: bn254ElementToHex(p.X),
		YBytesLE: bn254ElementToHex(p.Y),
		ZBytesLE: bn254ElementToHex(p.Z),
	}
}

func bls12381JacToJSON(p *gnarkbls12381.G1Jac) JacobianPointJSON {
	return JacobianPointJSON{
		XBytesLE: bls12381ElementToHex(p.X),
		YBytesLE: bls12381ElementToHex(p.Y),
		ZBytesLE: bls12381ElementToHex(p.Z),
	}
}
