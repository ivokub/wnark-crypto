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

type g1Case struct {
	Name                string      `json:"name"`
	PAffine             affinePoint `json:"p_affine"`
	QAffine             affinePoint `json:"q_affine"`
	PJacobian           jacPoint    `json:"p_jacobian"`
	PAffineOutput       jacPoint    `json:"p_affine_output"`
	NegPJacobian        jacPoint    `json:"neg_p_jacobian"`
	DoublePJacobian     jacPoint    `json:"double_p_jacobian"`
	AddMixedPPlusQJacob jacPoint    `json:"add_mixed_p_plus_q_jacobian"`
	AffineAddPPlusQ     jacPoint    `json:"affine_add_p_plus_q"`
}

type phase6Vectors struct {
	PointCases []g1Case `json:"point_cases"`
}

func main() {
	out := flag.String("out", "testdata/vectors/g1/bn254_phase6_g1_ops.json", "output vector path")
	flag.Parse()

	if err := run(*out); err != nil {
		panic(err)
	}
}

func run(outPath string) error {
	_, _, genAff, _ := gnarkbn254.Generators()

	infAff := new(gnarkbn254.G1Affine).SetInfinity()
	negGen := new(gnarkbn254.G1Affine).Neg(&genAff)
	five := scalarMul(5)
	seventeen := scalarMul(17)
	oneTwentyThree := scalarMul(123)

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

	out := phase6Vectors{
		PointCases: make([]g1Case, len(cases)),
	}
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

		out.PointCases[i] = g1Case{
			Name:                tc.name,
			PAffine:             affineToJSON(tc.p),
			QAffine:             affineToJSON(tc.q),
			PJacobian:           jacToJSON(&pJac),
			PAffineOutput:       affineOutputToJSON(&pAffineOut),
			NegPJacobian:        jacToJSON(&negP),
			DoublePJacobian:     jacToJSON(&doubleP),
			AddMixedPPlusQJacob: jacToJSON(&addMixed),
			AffineAddPPlusQ:     affineOutputToJSON(&affineAdd),
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

func scalarMul(v uint64) *gnarkbn254.G1Affine {
	_, _, genAff, _ := gnarkbn254.Generators()
	var out gnarkbn254.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func affineToJSON(p *gnarkbn254.G1Affine) affinePoint {
	return affinePoint{
		XBytesLE: elementToHex(p.X),
		YBytesLE: elementToHex(p.Y),
	}
}

func jacToJSON(p *gnarkbn254.G1Jac) jacPoint {
	return jacPoint{
		XBytesLE: elementToHex(p.X),
		YBytesLE: elementToHex(p.Y),
		ZBytesLE: elementToHex(p.Z),
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
