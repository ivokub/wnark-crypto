package testgen

import (
	"encoding/hex"
	"math/big"
	"math/rand"

	gnarkbls12377 "github.com/consensys/gnark-crypto/ecc/bls12-377"
	gnarkbls12377fp "github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	gnarkbls12377fr "github.com/consensys/gnark-crypto/ecc/bls12-377/fr"
)

const (
	BLS12377FpBytes      = 48
	BLS12377PointBytes   = 144
	BLS12377G2PointBytes = 288
)

func BuildRandomBLS12377G1Bases(count int, seed int64) ([]byte, error) {
	_, _, genAff, _ := gnarkbls12377.Generators()
	oneMontZ := montOneBLS12377()
	rng := rand.New(rand.NewSource(seed))
	scalars := make([]gnarkbls12377fr.Element, count)
	for i := range scalars {
		var raw [32]byte
		for j := range raw {
			raw[j] = byte(rng.Uint32())
		}
		scalars[i].SetBytes(raw[:])
		if scalars[i].IsZero() {
			scalars[i].SetUint64(1)
		}
	}
	points := gnarkbls12377.BatchScalarMultiplicationG1(&genAff, scalars)

	out := make([]byte, count*BLS12377PointBytes)
	for i := range points {
		base := i * BLS12377PointBytes
		writeElementLE6BLS12377(out[base:base+BLS12377FpBytes], points[i].X)
		writeElementLE6BLS12377(out[base+BLS12377FpBytes:base+2*BLS12377FpBytes], points[i].Y)
		writeElementLE6BLS12377(out[base+2*BLS12377FpBytes:base+3*BLS12377FpBytes], oneMontZ)
	}
	return out, nil
}

func BuildSequentialBLS12377G1Bases(count int) ([]byte, error) {
	_, _, genAff, _ := gnarkbls12377.Generators()
	oneMontZ := montOneBLS12377()
	scalars := make([]gnarkbls12377fr.Element, count)
	for i := range scalars {
		scalars[i].SetUint64(uint64(i + 1))
	}
	points := gnarkbls12377.BatchScalarMultiplicationG1(&genAff, scalars)

	out := make([]byte, count*BLS12377PointBytes)
	for i := range points {
		base := i * BLS12377PointBytes
		writeElementLE6BLS12377(out[base:base+BLS12377FpBytes], points[i].X)
		writeElementLE6BLS12377(out[base+BLS12377FpBytes:base+2*BLS12377FpBytes], points[i].Y)
		writeElementLE6BLS12377(out[base+2*BLS12377FpBytes:base+3*BLS12377FpBytes], oneMontZ)
	}
	return out, nil
}

func BuildSequentialBLS12377G2Bases(count int) ([]byte, error) {
	_, _, _, genAff := gnarkbls12377.Generators()
	oneMontZ := montOneBLS12377()
	zero := gnarkbls12377fp.Element{}
	scalars := make([]gnarkbls12377fr.Element, count)
	for i := range scalars {
		scalars[i].SetUint64(uint64(i + 1))
	}
	points := gnarkbls12377.BatchScalarMultiplicationG2(&genAff, scalars)

	out := make([]byte, count*BLS12377G2PointBytes)
	for i := range points {
		base := i * BLS12377G2PointBytes
		writeElementLE6BLS12377(out[base:base+BLS12377FpBytes], points[i].X.A0)
		writeElementLE6BLS12377(out[base+BLS12377FpBytes:base+2*BLS12377FpBytes], points[i].X.A1)
		writeElementLE6BLS12377(out[base+2*BLS12377FpBytes:base+3*BLS12377FpBytes], points[i].Y.A0)
		writeElementLE6BLS12377(out[base+3*BLS12377FpBytes:base+4*BLS12377FpBytes], points[i].Y.A1)
		writeElementLE6BLS12377(out[base+4*BLS12377FpBytes:base+5*BLS12377FpBytes], oneMontZ)
		writeElementLE6BLS12377(out[base+5*BLS12377FpBytes:base+6*BLS12377FpBytes], zero)
	}
	return out, nil
}

func BuildBLS12377G1BaseFixtureMetadata(count int) BaseFixtureMetadata {
	return BaseFixtureMetadata{
		Count:      count,
		PointBytes: BLS12377PointBytes,
		Format:     "jacobian_x_y_z_le",
	}
}

func BuildBLS12377G2BaseFixtureMetadata(count int) BaseFixtureMetadata {
	return BaseFixtureMetadata{
		Count:      count,
		PointBytes: BLS12377G2PointBytes,
		Format:     "jacobian_x_y_z_le",
	}
}

func BuildBLS12377G1OpsVectors() G1OpsVectorsJSON {
	_, _, genAff, _ := gnarkbls12377.Generators()
	infAff := new(gnarkbls12377.G1Affine).SetInfinity()
	negGen := new(gnarkbls12377.G1Affine).Neg(&genAff)
	five := scalarMulBLS12377G1Ops(5)
	seventeen := scalarMulBLS12377G1Ops(17)
	oneTwentyThree := scalarMulBLS12377G1Ops(123)

	cases := []struct {
		name string
		p    *gnarkbls12377.G1Affine
		q    *gnarkbls12377.G1Affine
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
		var pJac gnarkbls12377.G1Jac
		pJac.FromAffine(tc.p)

		var negP gnarkbls12377.G1Jac
		negP.Neg(&pJac)

		var doubleP gnarkbls12377.G1Jac
		doubleP.Double(&pJac)

		var addMixed gnarkbls12377.G1Jac
		addMixed.Set(&pJac).AddMixed(tc.q)

		var pAffineOut gnarkbls12377.G1Affine
		pAffineOut.FromJacobian(&pJac)

		var affineAdd gnarkbls12377.G1Affine
		affineAdd.Add(tc.p, tc.q)

		out.PointCases[i] = G1OpsCaseJSON{
			Name:                tc.name,
			PAffine:             bls12377AffineToJSON(tc.p),
			QAffine:             bls12377AffineToJSON(tc.q),
			PJacobian:           bls12377JacToJSON(&pJac),
			PAffineOutput:       bls12377AffineOutputToJSON(&pAffineOut),
			NegPJacobian:        bls12377JacToJSON(&negP),
			DoublePJacobian:     bls12377JacToJSON(&doubleP),
			AddMixedPPlusQJacob: bls12377JacToJSON(&addMixed),
			AffineAddPPlusQ:     bls12377AffineOutputToJSON(&affineAdd),
		}
	}
	return out
}

func BuildBLS12377G2OpsVectors() G2OpsVectorsJSON {
	_, _, _, genAff := gnarkbls12377.Generators()
	infAff := new(gnarkbls12377.G2Affine).SetInfinity()
	negGen := new(gnarkbls12377.G2Affine).Neg(&genAff)
	five := scalarMulBLS12377G2Ops(5)
	seventeen := scalarMulBLS12377G2Ops(17)
	oneTwentyThree := scalarMulBLS12377G2Ops(123)

	cases := []struct {
		name string
		p    *gnarkbls12377.G2Affine
		q    *gnarkbls12377.G2Affine
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
		var pJac gnarkbls12377.G2Jac
		pJac.FromAffine(tc.p)

		var negP gnarkbls12377.G2Jac
		negP.Neg(&pJac)

		var doubleP gnarkbls12377.G2Jac
		doubleP.Double(&pJac)

		var addMixed gnarkbls12377.G2Jac
		addMixed.Set(&pJac).AddMixed(tc.q)

		var pAffineOut gnarkbls12377.G2Affine
		pAffineOut.FromJacobian(&pJac)

		var affineAdd gnarkbls12377.G2Affine
		affineAdd.Add(tc.p, tc.q)

		out.PointCases[i] = G2OpsCaseJSON{
			Name:                tc.name,
			PAffine:             bls12377G2AffineToJSON(tc.p),
			QAffine:             bls12377G2AffineToJSON(tc.q),
			PJacobian:           bls12377G2JacToJSON(&pJac),
			PAffineOutput:       bls12377G2AffineOutputToJSON(&pAffineOut),
			NegPJacobian:        bls12377G2JacToJSON(&negP),
			DoublePJacobian:     bls12377G2JacToJSON(&doubleP),
			AddMixedPPlusQJacob: bls12377G2JacToJSON(&addMixed),
			AffineAddPPlusQ:     bls12377G2AffineOutputToJSON(&affineAdd),
		}
	}
	return out
}

func BuildBLS12377G1ScalarVectors() G1ScalarMulVectorsJSON {
	_, _, genAff, _ := gnarkbls12377.Generators()
	infAff := new(gnarkbls12377.G1Affine).SetInfinity()
	fiveGen := scalarMulBLS12377Generator(newBLS12377ScalarUint64(5))
	oneTwentyThreeGen := scalarMulBLS12377Generator(newBLS12377ScalarUint64(123))
	modMinusOne := new(big.Int).Sub(gnarkbls12377fr.Modulus(), big.NewInt(1))

	scalarCases := []struct {
		name   string
		base   *gnarkbls12377.G1Affine
		scalar gnarkbls12377fr.Element
	}{
		{name: "gen_times_zero", base: &genAff, scalar: newBLS12377ScalarUint64(0)},
		{name: "gen_times_one", base: &genAff, scalar: newBLS12377ScalarUint64(1)},
		{name: "gen_times_two", base: &genAff, scalar: newBLS12377ScalarUint64(2)},
		{name: "five_gen_times_seventeen", base: fiveGen, scalar: newBLS12377ScalarUint64(17)},
		{name: "infinity_times_one_twenty_three", base: infAff, scalar: newBLS12377ScalarUint64(123)},
		{name: "one_twenty_three_times_forty_two", base: oneTwentyThreeGen, scalar: newBLS12377ScalarUint64(42)},
		{name: "gen_times_q_minus_one", base: &genAff, scalar: newBLS12377ScalarBig(modMinusOne)},
	}

	baseCases := []struct {
		name   string
		scalar gnarkbls12377fr.Element
	}{
		{name: "base_zero", scalar: newBLS12377ScalarUint64(0)},
		{name: "base_one", scalar: newBLS12377ScalarUint64(1)},
		{name: "base_two", scalar: newBLS12377ScalarUint64(2)},
		{name: "base_one_twenty_three", scalar: newBLS12377ScalarUint64(123)},
		{name: "base_q_minus_one", scalar: newBLS12377ScalarBig(modMinusOne)},
	}

	out := G1ScalarMulVectorsJSON{
		GeneratorAffine: bls12377AffineToJSON(&genAff),
		OneMontZ:        bls12377ElementToHex(montOneBLS12377()),
		ScalarCases:     make([]G1ScalarMulCaseJSON, len(scalarCases)),
		BaseCases:       make([]G1ScalarMulBaseCaseJSON, len(baseCases)),
	}

	for i, tc := range scalarCases {
		var expected gnarkbls12377.G1Affine
		expected.ScalarMultiplication(tc.base, bls12377ScalarToBig(tc.scalar))
		out.ScalarCases[i] = G1ScalarMulCaseJSON{
			Name:            tc.name,
			BaseAffine:      bls12377AffineToJSON(tc.base),
			ScalarBytesLE:   bls12377ScalarToHex(tc.scalar),
			ScalarMulAffine: bls12377AffineOutputToJSON(&expected),
		}
	}

	for i, tc := range baseCases {
		var expected gnarkbls12377.G1Affine
		expected.ScalarMultiplicationBase(bls12377ScalarToBig(tc.scalar))
		out.BaseCases[i] = G1ScalarMulBaseCaseJSON{
			Name:                tc.name,
			ScalarBytesLE:       bls12377ScalarToHex(tc.scalar),
			ScalarMulBaseAffine: bls12377AffineOutputToJSON(&expected),
		}
	}

	return out
}

func BuildBLS12377G1MSMVectors() G1MSMVectorsJSON {
	_, _, genAff, _ := gnarkbls12377.Generators()
	infAff := new(gnarkbls12377.G1Affine).SetInfinity()
	five := scalarMulBLS12377G1MSM(5)
	seventeen := scalarMulBLS12377G1MSM(17)
	oneTwentyThree := scalarMulBLS12377G1MSM(123)
	twoHundredEleven := scalarMulBLS12377G1MSM(211)
	modMinusOne := new(big.Int).Sub(gnarkbls12377fr.Modulus(), big.NewInt(1))

	cases := []struct {
		name    string
		bases   []*gnarkbls12377.G1Affine
		scalars []gnarkbls12377fr.Element
	}{
		{
			name:    "single_generator",
			bases:   []*gnarkbls12377.G1Affine{&genAff, infAff, infAff, infAff},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarUint64(1), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(0)},
		},
		{
			name:    "simple_linear_combo",
			bases:   []*gnarkbls12377.G1Affine{&genAff, five, seventeen, oneTwentyThree},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarUint64(3), bls12377MSMScalarUint64(4), bls12377MSMScalarUint64(5), bls12377MSMScalarUint64(6)},
		},
		{
			name:    "includes_infinity_and_zero_scalar",
			bases:   []*gnarkbls12377.G1Affine{infAff, five, infAff, seventeen},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarUint64(19), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(7), bls12377MSMScalarUint64(9)},
		},
		{
			name:    "q_minus_one_mix",
			bases:   []*gnarkbls12377.G1Affine{&genAff, five, twoHundredEleven, oneTwentyThree},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarBig(modMinusOne), bls12377MSMScalarUint64(2), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(13)},
		},
	}

	out := G1MSMVectorsJSON{
		TermsPerInstance: 4,
		MSMCases:         make([]G1MSMCaseJSON, len(cases)),
		OneMontZ:         bls12377ElementToHex(montOneBLS12377()),
	}

	for i, tc := range cases {
		expected := naiveBLS12377MSM(tc.bases, tc.scalars)
		out.MSMCases[i] = G1MSMCaseJSON{
			Name:           tc.name,
			BasesAffine:    encodeBLS12377AffineBatch(tc.bases),
			ScalarsBytesLE: encodeBLS12377ScalarBatch(tc.scalars),
			ExpectedAffine: bls12377AffineOutputToJSON(expected),
		}
	}

	return out
}

func BuildBLS12377G2MSMVectors() G2MSMVectorsJSON {
	_, _, _, genAff := gnarkbls12377.Generators()
	infAff := new(gnarkbls12377.G2Affine).SetInfinity()
	five := scalarMulBLS12377G2Ops(5)
	seventeen := scalarMulBLS12377G2Ops(17)
	oneTwentyThree := scalarMulBLS12377G2Ops(123)
	twoHundredEleven := scalarMulBLS12377G2Ops(211)
	modMinusOne := new(big.Int).Sub(gnarkbls12377fr.Modulus(), big.NewInt(1))

	cases := []struct {
		name    string
		bases   []*gnarkbls12377.G2Affine
		scalars []gnarkbls12377fr.Element
	}{
		{
			name:    "single_generator",
			bases:   []*gnarkbls12377.G2Affine{&genAff, infAff, infAff, infAff},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarUint64(1), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(0)},
		},
		{
			name:    "simple_linear_combo",
			bases:   []*gnarkbls12377.G2Affine{&genAff, five, seventeen, oneTwentyThree},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarUint64(3), bls12377MSMScalarUint64(4), bls12377MSMScalarUint64(5), bls12377MSMScalarUint64(6)},
		},
		{
			name:    "includes_infinity_and_zero_scalar",
			bases:   []*gnarkbls12377.G2Affine{infAff, five, infAff, seventeen},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarUint64(19), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(7), bls12377MSMScalarUint64(9)},
		},
		{
			name:    "q_minus_one_mix",
			bases:   []*gnarkbls12377.G2Affine{&genAff, five, twoHundredEleven, oneTwentyThree},
			scalars: []gnarkbls12377fr.Element{bls12377MSMScalarBig(modMinusOne), bls12377MSMScalarUint64(2), bls12377MSMScalarUint64(0), bls12377MSMScalarUint64(13)},
		},
	}

	out := G2MSMVectorsJSON{
		TermsPerInstance: 4,
		MSMCases:         make([]G2MSMCaseJSON, len(cases)),
	}

	for i, tc := range cases {
		expected := naiveBLS12377G2MSM(tc.bases, tc.scalars)
		out.MSMCases[i] = G2MSMCaseJSON{
			Name:           tc.name,
			BasesAffine:    encodeBLS12377G2AffineBatch(tc.bases),
			ScalarsBytesLE: encodeBLS12377ScalarBatch(tc.scalars),
			ExpectedAffine: bls12377G2AffineOutputToJSON(expected),
		}
	}
	return out
}

func scalarMulBLS12377G1Ops(v uint64) *gnarkbls12377.G1Affine {
	_, _, genAff, _ := gnarkbls12377.Generators()
	var out gnarkbls12377.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func scalarMulBLS12377G2Ops(v uint64) *gnarkbls12377.G2Affine {
	_, _, _, genAff := gnarkbls12377.Generators()
	var out gnarkbls12377.G2Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func scalarMulBLS12377Generator(s gnarkbls12377fr.Element) *gnarkbls12377.G1Affine {
	_, _, genAff, _ := gnarkbls12377.Generators()
	var out gnarkbls12377.G1Affine
	out.ScalarMultiplication(&genAff, bls12377ScalarToBig(s))
	return &out
}

func scalarMulBLS12377G1MSM(v uint64) *gnarkbls12377.G1Affine {
	_, _, genAff, _ := gnarkbls12377.Generators()
	var out gnarkbls12377.G1Affine
	out.ScalarMultiplication(&genAff, new(big.Int).SetUint64(v))
	return &out
}

func newBLS12377ScalarUint64(v uint64) gnarkbls12377fr.Element {
	var out gnarkbls12377fr.Element
	out.SetUint64(v)
	return out
}

func newBLS12377ScalarBig(v *big.Int) gnarkbls12377fr.Element {
	var out gnarkbls12377fr.Element
	out.SetBigInt(v)
	return out
}

func bls12377MSMScalarUint64(v uint64) gnarkbls12377fr.Element {
	var out gnarkbls12377fr.Element
	out.SetUint64(v)
	return out
}

func bls12377MSMScalarBig(v *big.Int) gnarkbls12377fr.Element {
	var out gnarkbls12377fr.Element
	out.SetBigInt(v)
	return out
}

func bls12377ScalarToBig(v gnarkbls12377fr.Element) *big.Int {
	var out big.Int
	return v.BigInt(&out)
}

func bls12377ScalarToHex(v gnarkbls12377fr.Element) string {
	bytesBE := v.Bytes()
	return hex.EncodeToString(regularToLittleEndian(bytesBE[:]))
}

func bls12377AffineToJSON(p *gnarkbls12377.G1Affine) AffinePointJSON {
	return AffinePointJSON{
		XBytesLE: bls12377ElementToHex(p.X),
		YBytesLE: bls12377ElementToHex(p.Y),
	}
}

func bls12377JacToJSON(p *gnarkbls12377.G1Jac) JacobianPointJSON {
	return JacobianPointJSON{
		XBytesLE: bls12377ElementToHex(p.X),
		YBytesLE: bls12377ElementToHex(p.Y),
		ZBytesLE: bls12377ElementToHex(p.Z),
	}
}

func bls12377AffineOutputToJSON(p *gnarkbls12377.G1Affine) JacobianPointJSON {
	if p.IsInfinity() {
		return JacobianPointJSON{
			XBytesLE: zeroHexBLS12377(),
			YBytesLE: zeroHexBLS12377(),
			ZBytesLE: zeroHexBLS12377(),
		}
	}
	return JacobianPointJSON{
		XBytesLE: bls12377ElementToHex(p.X),
		YBytesLE: bls12377ElementToHex(p.Y),
		ZBytesLE: bls12377ElementToHex(montOneBLS12377()),
	}
}

func bls12377G2AffineToJSON(p *gnarkbls12377.G2Affine) G2AffinePointJSON {
	return G2AffinePointJSON{
		X: bls12377Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bls12377Fp2ToJSON(p.Y.A0, p.Y.A1),
	}
}

func bls12377G2JacToJSON(p *gnarkbls12377.G2Jac) G2JacobianPointJSON {
	return G2JacobianPointJSON{
		X: bls12377Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bls12377Fp2ToJSON(p.Y.A0, p.Y.A1),
		Z: bls12377Fp2ToJSON(p.Z.A0, p.Z.A1),
	}
}

func bls12377G2AffineOutputToJSON(p *gnarkbls12377.G2Affine) G2JacobianPointJSON {
	if p.IsInfinity() {
		return G2JacobianPointJSON{
			X: bls12377ZeroFp2JSON(),
			Y: bls12377ZeroFp2JSON(),
			Z: bls12377ZeroFp2JSON(),
		}
	}
	return G2JacobianPointJSON{
		X: bls12377Fp2ToJSON(p.X.A0, p.X.A1),
		Y: bls12377Fp2ToJSON(p.Y.A0, p.Y.A1),
		Z: bls12377OneFp2JSON(),
	}
}

func bls12377Fp2ToJSON(c0, c1 gnarkbls12377fp.Element) Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: bls12377ElementToHex(c0),
		C1BytesLE: bls12377ElementToHex(c1),
	}
}

func bls12377ZeroFp2JSON() Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: zeroHexBLS12377(),
		C1BytesLE: zeroHexBLS12377(),
	}
}

func bls12377OneFp2JSON() Fp2PointJSON {
	return Fp2PointJSON{
		C0BytesLE: bls12377ElementToHex(montOneBLS12377()),
		C1BytesLE: zeroHexBLS12377(),
	}
}

func encodeBLS12377AffineBatch(in []*gnarkbls12377.G1Affine) []AffinePointJSON {
	out := make([]AffinePointJSON, len(in))
	for i, point := range in {
		out[i] = bls12377AffineToJSON(point)
	}
	return out
}

func encodeBLS12377G2AffineBatch(in []*gnarkbls12377.G2Affine) []G2AffinePointJSON {
	out := make([]G2AffinePointJSON, len(in))
	for i, point := range in {
		out[i] = bls12377G2AffineToJSON(point)
	}
	return out
}

func encodeBLS12377ScalarBatch(in []gnarkbls12377fr.Element) []string {
	out := make([]string, len(in))
	for i, scalar := range in {
		out[i] = bls12377ScalarToHex(scalar)
	}
	return out
}

func naiveBLS12377MSM(bases []*gnarkbls12377.G1Affine, scalars []gnarkbls12377fr.Element) *gnarkbls12377.G1Affine {
	sum := new(gnarkbls12377.G1Affine).SetInfinity()
	for i := range bases {
		var term gnarkbls12377.G1Affine
		term.ScalarMultiplication(bases[i], bls12377ScalarToBig(scalars[i]))
		sum.Add(sum, &term)
	}
	return sum
}

func naiveBLS12377G2MSM(bases []*gnarkbls12377.G2Affine, scalars []gnarkbls12377fr.Element) *gnarkbls12377.G2Affine {
	sum := new(gnarkbls12377.G2Affine).SetInfinity()
	for i := range bases {
		var term gnarkbls12377.G2Affine
		term.ScalarMultiplication(bases[i], bls12377ScalarToBig(scalars[i]))
		sum.Add(sum, &term)
	}
	return sum
}

func bls12377ElementToHex(v gnarkbls12377fp.Element) string {
	return hex.EncodeToString(words6ToBytes(v[:]))
}

func montOneBLS12377() gnarkbls12377fp.Element {
	var one gnarkbls12377fp.Element
	one.SetOne()
	return one
}

func writeElementLE6BLS12377(dst []byte, v gnarkbls12377fp.Element) {
	for i, word := range [6]uint64(v) {
		base := i * 8
		dst[base+0] = byte(word)
		dst[base+1] = byte(word >> 8)
		dst[base+2] = byte(word >> 16)
		dst[base+3] = byte(word >> 24)
		dst[base+4] = byte(word >> 32)
		dst[base+5] = byte(word >> 40)
		dst[base+6] = byte(word >> 48)
		dst[base+7] = byte(word >> 56)
	}
}

func zeroHexBLS12377() string {
	return "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
}
