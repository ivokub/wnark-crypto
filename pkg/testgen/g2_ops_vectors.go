package testgen

import (
	"math/big"

	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkbls12381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkbn254fp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
)

type Fp2PointJSON struct {
	C0BytesLE string `json:"c0_bytes_le"`
	C1BytesLE string `json:"c1_bytes_le"`
}

type G2AffinePointJSON struct {
	X Fp2PointJSON `json:"x"`
	Y Fp2PointJSON `json:"y"`
}

type G2JacobianPointJSON struct {
	X Fp2PointJSON `json:"x"`
	Y Fp2PointJSON `json:"y"`
	Z Fp2PointJSON `json:"z"`
}

type G2OpsCaseJSON struct {
	Name                string              `json:"name"`
	PAffine             G2AffinePointJSON   `json:"p_affine"`
	QAffine             G2AffinePointJSON   `json:"q_affine"`
	PJacobian           G2JacobianPointJSON `json:"p_jacobian"`
	PAffineOutput       G2JacobianPointJSON `json:"p_affine_output"`
	NegPJacobian        G2JacobianPointJSON `json:"neg_p_jacobian"`
	DoublePJacobian     G2JacobianPointJSON `json:"double_p_jacobian"`
	AddMixedPPlusQJacob G2JacobianPointJSON `json:"add_mixed_p_plus_q_jacobian"`
	AffineAddPPlusQ     G2JacobianPointJSON `json:"affine_add_p_plus_q"`
}

type G2OpsVectorsJSON struct {
	PointCases []G2OpsCaseJSON `json:"point_cases"`
}

func BuildBN254G2OpsVectors() G2OpsVectorsJSON {
	_, _, _, genAff := gnarkbn254.Generators()
	infAff := new(gnarkbn254.G2Affine).SetInfinity()
	negGen := new(gnarkbn254.G2Affine).Neg(&genAff)
	five := scalarMulBN254G2Ops(5)
	seventeen := scalarMulBN254G2Ops(17)
	oneTwentyThree := scalarMulBN254G2Ops(123)

	cases := []struct {
		name string
		p    *gnarkbn254.G2Affine
		q    *gnarkbn254.G2Affine
	}{
		{name: "inf_plus_gen", p: infAff, q: &genAff},
		{name: "gen_plus_inf", p: &genAff, q: infAff},
		{name: "gen_plus_neg_gen", p: &genAff, q: negGen},
		{name: "gen_plus_gen", p: &genAff, q: &genAff},
		{name: "five_plus_seventeen", p: five, q: seventeen},
		{name: "one_twenty_three_self", p: oneTwentyThree, q: oneTwentyThree},
	}

	out := G2OpsVectorsJSON{PointCases: make([]G2OpsCaseJSON, len(cases))}
	for i, tc := range cases {
		var pJac gnarkbn254.G2Jac
		pJac.FromAffine(tc.p)

		var negP gnarkbn254.G2Jac
		negP.Neg(&pJac)

		var doubleP gnarkbn254.G2Jac
		doubleP.Double(&pJac)

		var addMixed gnarkbn254.G2Jac
		addMixed.Set(&pJac).AddMixed(tc.q)

		var pAffineOut gnarkbn254.G2Affine
		pAffineOut.FromJacobian(&pJac)

		var affineAdd gnarkbn254.G2Affine
		affineAdd.Add(tc.p, tc.q)

		out.PointCases[i] = G2OpsCaseJSON{
			Name:                tc.name,
			PAffine:             bn254G2AffineToJSON(tc.p),
			QAffine:             bn254G2AffineToJSON(tc.q),
			PJacobian:           bn254G2JacToJSON(&pJac),
			PAffineOutput:       bn254G2AffineOutputToJSON(&pAffineOut),
			NegPJacobian:        bn254G2JacToJSON(&negP),
			DoublePJacobian:     bn254G2JacToJSON(&doubleP),
			AddMixedPPlusQJacob: bn254G2JacToJSON(&addMixed),
			AffineAddPPlusQ:     bn254G2AffineOutputToJSON(&affineAdd),
		}
	}
	return out
}

func BuildBLS12381G2OpsVectors() G2OpsVectorsJSON {
	_, _, _, genAff := gnarkbls12381.Generators()
	infAff := new(gnarkbls12381.G2Affine).SetInfinity()
	negGen := new(gnarkbls12381.G2Affine).Neg(&genAff)
	five := scalarMulBLS12381G2Ops(5)
	seventeen := scalarMulBLS12381G2Ops(17)
	oneTwentyThree := scalarMulBLS12381G2Ops(123)

	cases := []struct {
		name string
		p    *gnarkbls12381.G2Affine
		q    *gnarkbls12381.G2Affine
	}{
		{name: "inf_plus_gen", p: infAff, q: &genAff},
		{name: "gen_plus_inf", p: &genAff, q: infAff},
		{name: "gen_plus_neg_gen", p: &genAff, q: negGen},
		{name: "gen_plus_gen", p: &genAff, q: &genAff},
		{name: "five_plus_seventeen", p: five, q: seventeen},
		{name: "one_twenty_three_self", p: oneTwentyThree, q: oneTwentyThree},
	}

	out := G2OpsVectorsJSON{PointCases: make([]G2OpsCaseJSON, len(cases))}
	for i, tc := range cases {
		var pJac gnarkbls12381.G2Jac
		pJac.FromAffine(tc.p)

		var negP gnarkbls12381.G2Jac
		negP.Neg(&pJac)

		var doubleP gnarkbls12381.G2Jac
		doubleP.Double(&pJac)

		var addMixed gnarkbls12381.G2Jac
		addMixed.Set(&pJac).AddMixed(tc.q)

		var pAffineOut gnarkbls12381.G2Affine
		pAffineOut.FromJacobian(&pJac)

		var affineAdd gnarkbls12381.G2Affine
		affineAdd.Add(tc.p, tc.q)

		out.PointCases[i] = G2OpsCaseJSON{
			Name:                tc.name,
			PAffine:             bls12381G2AffineToJSON(tc.p),
			QAffine:             bls12381G2AffineToJSON(tc.q),
			PJacobian:           bls12381G2JacToJSON(&pJac),
			PAffineOutput:       bls12381G2AffineOutputToJSON(&pAffineOut),
			NegPJacobian:        bls12381G2JacToJSON(&negP),
			DoublePJacobian:     bls12381G2JacToJSON(&doubleP),
			AddMixedPPlusQJacob: bls12381G2JacToJSON(&addMixed),
			AffineAddPPlusQ:     bls12381G2AffineOutputToJSON(&affineAdd),
		}
	}
	return out
}

func scalarMulBN254G2Ops(v uint64) *gnarkbn254.G2Affine {
	_, _, _, genAff := gnarkbn254.Generators()
	var out gnarkbn254.G2Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func scalarMulBLS12381G2Ops(v uint64) *gnarkbls12381.G2Affine {
	_, _, _, genAff := gnarkbls12381.Generators()
	var out gnarkbls12381.G2Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func bn254G2AffineToJSON(p *gnarkbn254.G2Affine) G2AffinePointJSON {
	return G2AffinePointJSON{
		X: bn254Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bn254Fp2ToJSON(p.Y.A0, p.Y.A1),
	}
}

func bls12381G2AffineToJSON(p *gnarkbls12381.G2Affine) G2AffinePointJSON {
	return G2AffinePointJSON{
		X: bls12381Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bls12381Fp2ToJSON(p.Y.A0, p.Y.A1),
	}
}

func bn254G2JacToJSON(p *gnarkbn254.G2Jac) G2JacobianPointJSON {
	return G2JacobianPointJSON{
		X: bn254Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bn254Fp2ToJSON(p.Y.A0, p.Y.A1),
		Z: bn254Fp2ToJSON(p.Z.A0, p.Z.A1),
	}
}

func bls12381G2JacToJSON(p *gnarkbls12381.G2Jac) G2JacobianPointJSON {
	return G2JacobianPointJSON{
		X: bls12381Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bls12381Fp2ToJSON(p.Y.A0, p.Y.A1),
		Z: bls12381Fp2ToJSON(p.Z.A0, p.Z.A1),
	}
}

func bn254G2AffineOutputToJSON(p *gnarkbn254.G2Affine) G2JacobianPointJSON {
	if p.IsInfinity() {
		return G2JacobianPointJSON{
			X: bn254ZeroFp2JSON(),
			Y: bn254ZeroFp2JSON(),
			Z: bn254ZeroFp2JSON(),
		}
	}
	return G2JacobianPointJSON{
		X: bn254Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bn254Fp2ToJSON(p.Y.A0, p.Y.A1),
		Z: bn254OneFp2JSON(),
	}
}

func bls12381G2AffineOutputToJSON(p *gnarkbls12381.G2Affine) G2JacobianPointJSON {
	if p.IsInfinity() {
		return G2JacobianPointJSON{
			X: bls12381ZeroFp2JSON(),
			Y: bls12381ZeroFp2JSON(),
			Z: bls12381ZeroFp2JSON(),
		}
	}
	return G2JacobianPointJSON{
		X: bls12381Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bls12381Fp2ToJSON(p.Y.A0, p.Y.A1),
		Z: bls12381OneFp2JSON(),
	}
}

func bn254Fp2ToJSON(c0, c1 gnarkbn254fp.Element) Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: bn254ElementToHex(c0),
		C1BytesLE: bn254ElementToHex(c1),
	}
}

func bls12381Fp2ToJSON(c0, c1 gnarkbls12381fp.Element) Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: bls12381ElementToHex(c0),
		C1BytesLE: bls12381ElementToHex(c1),
	}
}

func bn254ZeroFp2JSON() Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: zeroHexBN254(),
		C1BytesLE: zeroHexBN254(),
	}
}

func bls12381ZeroFp2JSON() Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: zeroHexBLS12381(),
		C1BytesLE: zeroHexBLS12381(),
	}
}

func bn254OneFp2JSON() Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: bn254ElementToHex(montOneBN254ScalarVectors()),
		C1BytesLE: zeroHexBN254(),
	}
}

func bls12381OneFp2JSON() Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: bls12381ElementToHex(montOneBLS12381ScalarVectors()),
		C1BytesLE: zeroHexBLS12381(),
	}
}
