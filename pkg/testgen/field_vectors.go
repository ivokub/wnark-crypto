package testgen

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math/big"
	"math/bits"
	"math/rand"

	gnarkbls12381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	gnarkbls12381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkbn254fp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	gnarkbn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type FieldElementCaseJSON struct {
	Name           string `json:"name"`
	ABytesLE       string `json:"a_bytes_le"`
	BBytesLE       string `json:"b_bytes_le"`
	EqualBytesLE   string `json:"equal_bytes_le"`
	AddBytesLE     string `json:"add_bytes_le"`
	SubBytesLE     string `json:"sub_bytes_le"`
	NegABytesLE    string `json:"neg_a_bytes_le"`
	DoubleABytesLE string `json:"double_a_bytes_le"`
	MulBytesLE     string `json:"mul_bytes_le"`
	SquareABytesLE string `json:"square_a_bytes_le"`
}

type NormalizeCaseJSON struct {
	Name            string `json:"name"`
	InputBytesLE    string `json:"input_bytes_le"`
	ExpectedBytesLE string `json:"expected_bytes_le"`
}

type ConvertCaseJSON struct {
	Name         string `json:"name"`
	RegularBytes string `json:"regular_bytes_le"`
	MontBytes    string `json:"mont_bytes_le"`
}

type FieldOpsVectorsJSON struct {
	ElementCases      []FieldElementCaseJSON `json:"element_cases"`
	EdgeCases         []FieldElementCaseJSON `json:"edge_cases"`
	DifferentialCases []FieldElementCaseJSON `json:"differential_cases"`
	NormalizeCases    []NormalizeCaseJSON    `json:"normalize_cases"`
	ConvertCases      []ConvertCaseJSON      `json:"convert_cases"`
}

type BN254VectorCaseJSON struct {
	Name               string   `json:"name"`
	RegularInputs      []string `json:"regular_inputs_le"`
	MontInputs         []string `json:"mont_inputs_le"`
	MontFactors        []string `json:"mont_factors_le"`
	AddExpected        []string `json:"add_expected_le"`
	SubExpected        []string `json:"sub_expected_le"`
	MulExpected        []string `json:"mul_expected_le"`
	ToMontExpected     []string `json:"to_mont_expected_le"`
	FromMontExpected   []string `json:"from_mont_expected_le"`
	BitReverseExpected []string `json:"bit_reverse_expected_le"`
}

type BN254VectorOpsJSON struct {
	VectorCases []BN254VectorCaseJSON `json:"vector_cases"`
}

func BuildBN254FROpsVectors() FieldOpsVectorsJSON {
	rng := rand.New(rand.NewSource(20260403))
	return FieldOpsVectorsJSON{
		ElementCases: []FieldElementCaseJSON{
			buildBN254FRElementCase("zero_zero", bn254RegularUint64(0), bn254RegularUint64(0)),
			buildBN254FRElementCase("zero_one", bn254RegularUint64(0), bn254RegularUint64(1)),
			buildBN254FRElementCase("one_one", bn254RegularUint64(1), bn254RegularUint64(1)),
			buildBN254FRElementCase("two_five", bn254RegularUint64(2), bn254RegularUint64(5)),
			buildBN254FRElementCase("neg_one_one", bn254FRQMinusOne(), bn254RegularUint64(1)),
			buildBN254FRElementCase("seven_five", bn254RegularUint64(7), bn254RegularUint64(5)),
		},
		EdgeCases: []FieldElementCaseJSON{
			buildBN254FRElementCase("carry32_plus_one", bn254RegularPow2MinusOne(32), bn254RegularUint64(1)),
			buildBN254FRElementCase("carry64_plus_one", bn254RegularPow2MinusOne(64), bn254RegularUint64(1)),
			buildBN254FRElementCase("carry128_plus_one", bn254RegularPow2MinusOne(128), bn254RegularUint64(1)),
			buildBN254FRElementCase("carry192_plus_one", bn254RegularPow2MinusOne(192), bn254RegularUint64(1)),
			buildBN254FRElementCase("q_minus_two_plus_three", bn254FRQMinus(2), bn254RegularUint64(3)),
			buildBN254FRElementCase("q_minus_one_q_minus_one", bn254FRQMinusOne(), bn254FRQMinusOne()),
			buildBN254FRElementCase("q_minus_two_q_minus_one", bn254FRQMinus(2), bn254FRQMinusOne()),
			buildBN254FRElementCase("half_q_floor_half_q_ceil", bn254FRFloorHalfModulus(), bn254FRCeilHalfModulus()),
		},
		DifferentialCases: buildBN254FRDifferentialCases(rng, 32),
		NormalizeCases: []NormalizeCaseJSON{
			{Name: "zero", InputBytesLE: bn254FRRegularHex(bn254RegularUint64(0)), ExpectedBytesLE: bn254FRRegularHex(bn254RegularUint64(0))},
			{Name: "one", InputBytesLE: bn254FRRegularHex(bn254RegularUint64(1)), ExpectedBytesLE: bn254FRRegularHex(bn254RegularUint64(1))},
			{Name: "q_minus_one", InputBytesLE: bn254FRRegularHex(bn254FRQMinusOne()), ExpectedBytesLE: bn254FRRegularHex(bn254FRQMinusOne())},
			{Name: "q", InputBytesLE: bn254FRRegularHex(bn254FRModulus()), ExpectedBytesLE: bn254FRRegularHex(bn254RegularUint64(0))},
			{Name: "q_plus_one", InputBytesLE: bn254FRRegularHex(addBig(bn254FRModulus(), bn254RegularUint64(1))), ExpectedBytesLE: bn254FRRegularHex(bn254RegularUint64(1))},
			{Name: "two_q_minus_one", InputBytesLE: bn254FRRegularHex(subBig(mulBig(bn254FRModulus(), bn254RegularUint64(2)), bn254RegularUint64(1))), ExpectedBytesLE: bn254FRRegularHex(bn254FRQMinusOne())},
		},
		ConvertCases: []ConvertCaseJSON{
			buildBN254FRConvertCase("zero", bn254RegularUint64(0)),
			buildBN254FRConvertCase("one", bn254RegularUint64(1)),
			buildBN254FRConvertCase("two", bn254RegularUint64(2)),
			buildBN254FRConvertCase("five", bn254RegularUint64(5)),
			buildBN254FRConvertCase("seven", bn254RegularUint64(7)),
			buildBN254FRConvertCase("q_minus_one", bn254FRQMinusOne()),
		},
	}
}

func BuildBLS12381FROpsVectors() FieldOpsVectorsJSON {
	rng := rand.New(rand.NewSource(20260404))
	return FieldOpsVectorsJSON{
		ElementCases: []FieldElementCaseJSON{
			buildBLS12381FRElementCase("zero_zero", blsRegularUint64(0), blsRegularUint64(0)),
			buildBLS12381FRElementCase("zero_one", blsRegularUint64(0), blsRegularUint64(1)),
			buildBLS12381FRElementCase("one_one", blsRegularUint64(1), blsRegularUint64(1)),
			buildBLS12381FRElementCase("two_five", blsRegularUint64(2), blsRegularUint64(5)),
			buildBLS12381FRElementCase("neg_one_one", blsFRQMinusOne(), blsRegularUint64(1)),
			buildBLS12381FRElementCase("seven_five", blsRegularUint64(7), blsRegularUint64(5)),
		},
		EdgeCases: []FieldElementCaseJSON{
			buildBLS12381FRElementCase("carry32_plus_one", blsRegularPow2MinusOne(32), blsRegularUint64(1)),
			buildBLS12381FRElementCase("carry64_plus_one", blsRegularPow2MinusOne(64), blsRegularUint64(1)),
			buildBLS12381FRElementCase("carry128_plus_one", blsRegularPow2MinusOne(128), blsRegularUint64(1)),
			buildBLS12381FRElementCase("carry192_plus_one", blsRegularPow2MinusOne(192), blsRegularUint64(1)),
			buildBLS12381FRElementCase("q_minus_two_plus_three", blsFRQMinus(2), blsRegularUint64(3)),
			buildBLS12381FRElementCase("q_minus_one_q_minus_one", blsFRQMinusOne(), blsFRQMinusOne()),
			buildBLS12381FRElementCase("q_minus_two_q_minus_one", blsFRQMinus(2), blsFRQMinusOne()),
			buildBLS12381FRElementCase("half_q_floor_half_q_ceil", blsFRFloorHalfModulus(), blsFRCeilHalfModulus()),
		},
		DifferentialCases: buildBLS12381FRDifferentialCases(rng, 32),
		NormalizeCases: []NormalizeCaseJSON{
			{Name: "zero", InputBytesLE: blsFRRegularHex(blsRegularUint64(0)), ExpectedBytesLE: blsFRRegularHex(blsRegularUint64(0))},
			{Name: "one", InputBytesLE: blsFRRegularHex(blsRegularUint64(1)), ExpectedBytesLE: blsFRRegularHex(blsRegularUint64(1))},
			{Name: "q_minus_one", InputBytesLE: blsFRRegularHex(blsFRQMinusOne()), ExpectedBytesLE: blsFRRegularHex(blsFRQMinusOne())},
			{Name: "q", InputBytesLE: blsFRRegularHex(blsFRModulus()), ExpectedBytesLE: blsFRRegularHex(blsRegularUint64(0))},
			{Name: "q_plus_one", InputBytesLE: blsFRRegularHex(addBig(blsFRModulus(), blsRegularUint64(1))), ExpectedBytesLE: blsFRRegularHex(blsRegularUint64(1))},
			{Name: "two_q_minus_one", InputBytesLE: blsFRRegularHex(subBig(mulBig(blsFRModulus(), blsRegularUint64(2)), blsRegularUint64(1))), ExpectedBytesLE: blsFRRegularHex(blsFRQMinusOne())},
		},
		ConvertCases: []ConvertCaseJSON{
			buildBLS12381FRConvertCase("zero", blsRegularUint64(0)),
			buildBLS12381FRConvertCase("one", blsRegularUint64(1)),
			buildBLS12381FRConvertCase("two", blsRegularUint64(2)),
			buildBLS12381FRConvertCase("five", blsRegularUint64(5)),
			buildBLS12381FRConvertCase("seven", blsRegularUint64(7)),
			buildBLS12381FRConvertCase("q_minus_one", blsFRQMinusOne()),
		},
	}
}

func BuildBN254FPOpsVectors() FieldOpsVectorsJSON {
	rng := rand.New(rand.NewSource(20260402))
	return FieldOpsVectorsJSON{
		ElementCases: []FieldElementCaseJSON{
			buildBN254FPElementCase("zero_zero", bn254RegularUint64(0), bn254RegularUint64(0)),
			buildBN254FPElementCase("zero_one", bn254RegularUint64(0), bn254RegularUint64(1)),
			buildBN254FPElementCase("one_one", bn254RegularUint64(1), bn254RegularUint64(1)),
			buildBN254FPElementCase("two_five", bn254RegularUint64(2), bn254RegularUint64(5)),
			buildBN254FPElementCase("neg_one_one", bn254FPQMinusOne(), bn254RegularUint64(1)),
			buildBN254FPElementCase("seven_five", bn254RegularUint64(7), bn254RegularUint64(5)),
		},
		EdgeCases: []FieldElementCaseJSON{
			buildBN254FPElementCase("carry32_plus_one", bn254RegularPow2MinusOne(32), bn254RegularUint64(1)),
			buildBN254FPElementCase("carry64_plus_one", bn254RegularPow2MinusOne(64), bn254RegularUint64(1)),
			buildBN254FPElementCase("carry128_plus_one", bn254RegularPow2MinusOne(128), bn254RegularUint64(1)),
			buildBN254FPElementCase("carry192_plus_one", bn254RegularPow2MinusOne(192), bn254RegularUint64(1)),
			buildBN254FPElementCase("q_minus_two_plus_three", bn254FPQMinus(2), bn254RegularUint64(3)),
			buildBN254FPElementCase("q_minus_one_q_minus_one", bn254FPQMinusOne(), bn254FPQMinusOne()),
			buildBN254FPElementCase("q_minus_two_q_minus_one", bn254FPQMinus(2), bn254FPQMinusOne()),
			buildBN254FPElementCase("half_q_floor_half_q_ceil", bn254FPFloorHalfModulus(), bn254FPCeilHalfModulus()),
		},
		DifferentialCases: buildBN254FPDifferentialCases(rng, 32),
		NormalizeCases: []NormalizeCaseJSON{
			{Name: "zero", InputBytesLE: bn254FPRegularHex(bn254RegularUint64(0)), ExpectedBytesLE: bn254FPRegularHex(bn254RegularUint64(0))},
			{Name: "one", InputBytesLE: bn254FPRegularHex(bn254RegularUint64(1)), ExpectedBytesLE: bn254FPRegularHex(bn254RegularUint64(1))},
			{Name: "q_minus_one", InputBytesLE: bn254FPRegularHex(bn254FPQMinusOne()), ExpectedBytesLE: bn254FPRegularHex(bn254FPQMinusOne())},
			{Name: "q", InputBytesLE: bn254FPRegularHex(bn254FPModulus()), ExpectedBytesLE: bn254FPRegularHex(bn254RegularUint64(0))},
			{Name: "q_plus_one", InputBytesLE: bn254FPRegularHex(addBig(bn254FPModulus(), bn254RegularUint64(1))), ExpectedBytesLE: bn254FPRegularHex(bn254RegularUint64(1))},
			{Name: "two_q_minus_one", InputBytesLE: bn254FPRegularHex(subBig(mulBig(bn254FPModulus(), bn254RegularUint64(2)), bn254RegularUint64(1))), ExpectedBytesLE: bn254FPRegularHex(bn254FPQMinusOne())},
		},
		ConvertCases: []ConvertCaseJSON{
			buildBN254FPConvertCase("zero", bn254RegularUint64(0)),
			buildBN254FPConvertCase("one", bn254RegularUint64(1)),
			buildBN254FPConvertCase("two", bn254RegularUint64(2)),
			buildBN254FPConvertCase("five", bn254RegularUint64(5)),
			buildBN254FPConvertCase("seven", bn254RegularUint64(7)),
			buildBN254FPConvertCase("q_minus_one", bn254FPQMinusOne()),
		},
	}
}

func BuildBLS12381FPOpsVectors() FieldOpsVectorsJSON {
	rng := rand.New(rand.NewSource(20260405))
	return FieldOpsVectorsJSON{
		ElementCases: []FieldElementCaseJSON{
			buildBLS12381FPElementCase("zero_zero", blsRegularUint64(0), blsRegularUint64(0)),
			buildBLS12381FPElementCase("zero_one", blsRegularUint64(0), blsRegularUint64(1)),
			buildBLS12381FPElementCase("one_one", blsRegularUint64(1), blsRegularUint64(1)),
			buildBLS12381FPElementCase("two_five", blsRegularUint64(2), blsRegularUint64(5)),
			buildBLS12381FPElementCase("neg_one_one", blsFPQMinusOne(), blsRegularUint64(1)),
			buildBLS12381FPElementCase("seven_five", blsRegularUint64(7), blsRegularUint64(5)),
		},
		EdgeCases: []FieldElementCaseJSON{
			buildBLS12381FPElementCase("carry32_plus_one", blsRegularPow2MinusOne(32), blsRegularUint64(1)),
			buildBLS12381FPElementCase("carry64_plus_one", blsRegularPow2MinusOne(64), blsRegularUint64(1)),
			buildBLS12381FPElementCase("carry128_plus_one", blsRegularPow2MinusOne(128), blsRegularUint64(1)),
			buildBLS12381FPElementCase("carry192_plus_one", blsRegularPow2MinusOne(192), blsRegularUint64(1)),
			buildBLS12381FPElementCase("q_minus_two_plus_three", blsFPQMinus(2), blsRegularUint64(3)),
			buildBLS12381FPElementCase("q_minus_one_q_minus_one", blsFPQMinusOne(), blsFPQMinusOne()),
			buildBLS12381FPElementCase("q_minus_two_q_minus_one", blsFPQMinus(2), blsFPQMinusOne()),
			buildBLS12381FPElementCase("half_q_floor_half_q_ceil", blsFPFloorHalfModulus(), blsFPCeilHalfModulus()),
		},
		DifferentialCases: buildBLS12381FPDifferentialCases(rng, 32),
		NormalizeCases: []NormalizeCaseJSON{
			{Name: "zero", InputBytesLE: blsFPRegularHex(blsRegularUint64(0)), ExpectedBytesLE: blsFPRegularHex(blsRegularUint64(0))},
			{Name: "one", InputBytesLE: blsFPRegularHex(blsRegularUint64(1)), ExpectedBytesLE: blsFPRegularHex(blsRegularUint64(1))},
			{Name: "q_minus_one", InputBytesLE: blsFPRegularHex(blsFPQMinusOne()), ExpectedBytesLE: blsFPRegularHex(blsFPQMinusOne())},
			{Name: "q", InputBytesLE: blsFPRegularHex(blsFPModulus()), ExpectedBytesLE: blsFPRegularHex(blsRegularUint64(0))},
			{Name: "q_plus_one", InputBytesLE: blsFPRegularHex(addBig(blsFPModulus(), blsRegularUint64(1))), ExpectedBytesLE: blsFPRegularHex(blsRegularUint64(1))},
			{Name: "two_q_minus_one", InputBytesLE: blsFPRegularHex(subBig(mulBig(blsFPModulus(), blsRegularUint64(2)), blsRegularUint64(1))), ExpectedBytesLE: blsFPRegularHex(blsFPQMinusOne())},
		},
		ConvertCases: []ConvertCaseJSON{
			buildBLS12381FPConvertCase("zero", blsRegularUint64(0)),
			buildBLS12381FPConvertCase("one", blsRegularUint64(1)),
			buildBLS12381FPConvertCase("two", blsRegularUint64(2)),
			buildBLS12381FPConvertCase("five", blsRegularUint64(5)),
			buildBLS12381FPConvertCase("seven", blsRegularUint64(7)),
			buildBLS12381FPConvertCase("q_minus_one", blsFPQMinusOne()),
		},
	}
}

func BuildBN254FRVectorOps() BN254VectorOpsJSON {
	return BN254VectorOpsJSON{
		VectorCases: []BN254VectorCaseJSON{
			buildBN254FRVectorCase("n8_random", 8, rand.New(rand.NewSource(2026040201))),
			buildBN254FRVectorCase("n16_random", 16, rand.New(rand.NewSource(2026040202))),
		},
	}
}

func BuildBLS12381FRVectorOps() BN254VectorOpsJSON {
	return BN254VectorOpsJSON{
		VectorCases: []BN254VectorCaseJSON{
			buildBLS12381FRVectorCase("n8_random", 8, rand.New(rand.NewSource(2026040401))),
			buildBLS12381FRVectorCase("n16_random", 16, rand.New(rand.NewSource(2026040402))),
		},
	}
}

func buildBN254FRVectorCase(name string, size int, rng *rand.Rand) BN254VectorCaseJSON {
	out := BN254VectorCaseJSON{
		Name:               name,
		RegularInputs:      make([]string, size),
		MontInputs:         make([]string, size),
		MontFactors:        make([]string, size),
		AddExpected:        make([]string, size),
		SubExpected:        make([]string, size),
		MulExpected:        make([]string, size),
		ToMontExpected:     make([]string, size),
		FromMontExpected:   make([]string, size),
		BitReverseExpected: make([]string, size),
	}

	for i := 0; i < size; i++ {
		aRegular := randomBN254FRFieldBigInt(rng)
		bRegular := randomBN254FRFieldBigInt(rng)
		var aMont, bMont gnarkbn254fr.Element
		aMont.SetBigInt(aRegular)
		bMont.SetBigInt(bRegular)

		var add, sub, mul gnarkbn254fr.Element
		add.Add(&aMont, &bMont)
		sub.Sub(&aMont, &bMont)
		mul.Mul(&aMont, &bMont)

		out.RegularInputs[i] = bn254FRRegularHex(aRegular)
		out.MontInputs[i] = bn254FrElementHex(aMont)
		out.MontFactors[i] = bn254FrElementHex(bMont)
		out.AddExpected[i] = bn254FrElementHex(add)
		out.SubExpected[i] = bn254FrElementHex(sub)
		out.MulExpected[i] = bn254FrElementHex(mul)
		out.ToMontExpected[i] = bn254FrElementHex(aMont)
		out.FromMontExpected[i] = bn254FRRegularHex(aRegular)
	}

	logCount := bits.Len(uint(size)) - 1
	for i := 0; i < size; i++ {
		j := int(bits.Reverse64(uint64(i)) >> (64 - logCount))
		out.BitReverseExpected[i] = out.MontInputs[j]
	}

	return out
}

func buildBLS12381FRVectorCase(name string, size int, rng *rand.Rand) BN254VectorCaseJSON {
	out := BN254VectorCaseJSON{
		Name:               name,
		RegularInputs:      make([]string, size),
		MontInputs:         make([]string, size),
		MontFactors:        make([]string, size),
		AddExpected:        make([]string, size),
		SubExpected:        make([]string, size),
		MulExpected:        make([]string, size),
		ToMontExpected:     make([]string, size),
		FromMontExpected:   make([]string, size),
		BitReverseExpected: make([]string, size),
	}

	for i := 0; i < size; i++ {
		aRegular := randomBLS12381FRFieldBigInt(rng)
		bRegular := randomBLS12381FRFieldBigInt(rng)
		var aMont, bMont gnarkbls12381fr.Element
		aMont.SetBigInt(aRegular)
		bMont.SetBigInt(bRegular)

		var add, sub, mul gnarkbls12381fr.Element
		add.Add(&aMont, &bMont)
		sub.Sub(&aMont, &bMont)
		mul.Mul(&aMont, &bMont)

		out.RegularInputs[i] = blsFRRegularHex(aRegular)
		out.MontInputs[i] = bls12381FrElementHex(aMont)
		out.MontFactors[i] = bls12381FrElementHex(bMont)
		out.AddExpected[i] = bls12381FrElementHex(add)
		out.SubExpected[i] = bls12381FrElementHex(sub)
		out.MulExpected[i] = bls12381FrElementHex(mul)
		out.ToMontExpected[i] = bls12381FrElementHex(aMont)
		out.FromMontExpected[i] = blsFRRegularHex(aRegular)
	}

	logCount := bits.Len(uint(size)) - 1
	for i := 0; i < size; i++ {
		j := int(bits.Reverse64(uint64(i)) >> (64 - logCount))
		out.BitReverseExpected[i] = out.MontInputs[j]
	}

	return out
}

func buildBN254FRElementCase(name string, aRegular, bRegular *big.Int) FieldElementCaseJSON {
	aMont := bn254FRToMont(aRegular)
	bMont := bn254FRToMont(bRegular)
	var add, sub, negA, dblA, mul, sqA gnarkbn254fr.Element
	add.Add(&aMont, &bMont)
	sub.Sub(&aMont, &bMont)
	negA.Neg(&aMont)
	dblA.Double(&aMont)
	mul.Mul(&aMont, &bMont)
	sqA.Square(&aMont)
	equal := bn254FRZeroMont()
	if aMont.Equal(&bMont) {
		equal.SetUint64(1)
	}
	return FieldElementCaseJSON{
		Name:           name,
		ABytesLE:       bn254FrElementHex(aMont),
		BBytesLE:       bn254FrElementHex(bMont),
		EqualBytesLE:   bn254FrElementHex(equal),
		AddBytesLE:     bn254FrElementHex(add),
		SubBytesLE:     bn254FrElementHex(sub),
		NegABytesLE:    bn254FrElementHex(negA),
		DoubleABytesLE: bn254FrElementHex(dblA),
		MulBytesLE:     bn254FrElementHex(mul),
		SquareABytesLE: bn254FrElementHex(sqA),
	}
}

func buildBLS12381FRElementCase(name string, aRegular, bRegular *big.Int) FieldElementCaseJSON {
	aMont := bls12381FRToMont(aRegular)
	bMont := bls12381FRToMont(bRegular)
	var add, sub, negA, dblA, mul, sqA gnarkbls12381fr.Element
	add.Add(&aMont, &bMont)
	sub.Sub(&aMont, &bMont)
	negA.Neg(&aMont)
	dblA.Double(&aMont)
	mul.Mul(&aMont, &bMont)
	sqA.Square(&aMont)
	equal := bls12381FRZeroMont()
	if aMont.Equal(&bMont) {
		equal.SetUint64(1)
	}
	return FieldElementCaseJSON{
		Name:           name,
		ABytesLE:       bls12381FrElementHex(aMont),
		BBytesLE:       bls12381FrElementHex(bMont),
		EqualBytesLE:   bls12381FrElementHex(equal),
		AddBytesLE:     bls12381FrElementHex(add),
		SubBytesLE:     bls12381FrElementHex(sub),
		NegABytesLE:    bls12381FrElementHex(negA),
		DoubleABytesLE: bls12381FrElementHex(dblA),
		MulBytesLE:     bls12381FrElementHex(mul),
		SquareABytesLE: bls12381FrElementHex(sqA),
	}
}

func buildBN254FPElementCase(name string, aRegular, bRegular *big.Int) FieldElementCaseJSON {
	aMont := bn254FPToMont(aRegular)
	bMont := bn254FPToMont(bRegular)
	var add, sub, negA, dblA, mul, sqA gnarkbn254fp.Element
	add.Add(&aMont, &bMont)
	sub.Sub(&aMont, &bMont)
	negA.Neg(&aMont)
	dblA.Double(&aMont)
	mul.Mul(&aMont, &bMont)
	sqA.Square(&aMont)
	equal := bn254FPZeroMont()
	if aMont.Equal(&bMont) {
		equal.SetUint64(1)
	}
	return FieldElementCaseJSON{
		Name:           name,
		ABytesLE:       bn254FpElementHex(aMont),
		BBytesLE:       bn254FpElementHex(bMont),
		EqualBytesLE:   bn254FpElementHex(equal),
		AddBytesLE:     bn254FpElementHex(add),
		SubBytesLE:     bn254FpElementHex(sub),
		NegABytesLE:    bn254FpElementHex(negA),
		DoubleABytesLE: bn254FpElementHex(dblA),
		MulBytesLE:     bn254FpElementHex(mul),
		SquareABytesLE: bn254FpElementHex(sqA),
	}
}

func buildBLS12381FPElementCase(name string, aRegular, bRegular *big.Int) FieldElementCaseJSON {
	aMont := bls12381FPToMont(aRegular)
	bMont := bls12381FPToMont(bRegular)
	var add, sub, negA, dblA, mul, sqA gnarkbls12381fp.Element
	add.Add(&aMont, &bMont)
	sub.Sub(&aMont, &bMont)
	negA.Neg(&aMont)
	dblA.Double(&aMont)
	mul.Mul(&aMont, &bMont)
	sqA.Square(&aMont)
	equal := bls12381FPZeroMont()
	if aMont.Equal(&bMont) {
		equal.SetUint64(1)
	}
	return FieldElementCaseJSON{
		Name:           name,
		ABytesLE:       bls12381FpElementHex(aMont),
		BBytesLE:       bls12381FpElementHex(bMont),
		EqualBytesLE:   bls12381FpElementHex(equal),
		AddBytesLE:     bls12381FpElementHex(add),
		SubBytesLE:     bls12381FpElementHex(sub),
		NegABytesLE:    bls12381FpElementHex(negA),
		DoubleABytesLE: bls12381FpElementHex(dblA),
		MulBytesLE:     bls12381FpElementHex(mul),
		SquareABytesLE: bls12381FpElementHex(sqA),
	}
}

func buildBN254FRConvertCase(name string, regular *big.Int) ConvertCaseJSON {
	return ConvertCaseJSON{Name: name, RegularBytes: bn254FRRegularHex(regular), MontBytes: bn254FrElementHex(bn254FRToMont(regular))}
}

func buildBLS12381FRConvertCase(name string, regular *big.Int) ConvertCaseJSON {
	return ConvertCaseJSON{Name: name, RegularBytes: blsFRRegularHex(regular), MontBytes: bls12381FrElementHex(bls12381FRToMont(regular))}
}

func buildBN254FPConvertCase(name string, regular *big.Int) ConvertCaseJSON {
	return ConvertCaseJSON{Name: name, RegularBytes: bn254FPRegularHex(regular), MontBytes: bn254FpElementHex(bn254FPToMont(regular))}
}

func buildBLS12381FPConvertCase(name string, regular *big.Int) ConvertCaseJSON {
	return ConvertCaseJSON{Name: name, RegularBytes: blsFPRegularHex(regular), MontBytes: bls12381FpElementHex(bls12381FPToMont(regular))}
}

func buildBN254FRDifferentialCases(rng *rand.Rand, count int) []FieldElementCaseJSON {
	out := make([]FieldElementCaseJSON, count)
	for i := 0; i < count; i++ {
		a := randomBN254FRFieldBigInt(rng)
		b := randomBN254FRFieldBigInt(rng)
		if i%7 == 0 {
			b = new(big.Int).Set(a)
		}
		out[i] = buildBN254FRElementCase(fmt.Sprintf("random_%02d", i), a, b)
	}
	return out
}

func buildBLS12381FRDifferentialCases(rng *rand.Rand, count int) []FieldElementCaseJSON {
	out := make([]FieldElementCaseJSON, count)
	for i := 0; i < count; i++ {
		a := randomBLS12381FRFieldBigInt(rng)
		b := randomBLS12381FRFieldBigInt(rng)
		if i%7 == 0 {
			b = new(big.Int).Set(a)
		}
		out[i] = buildBLS12381FRElementCase(fmt.Sprintf("random_%02d", i), a, b)
	}
	return out
}

func buildBN254FPDifferentialCases(rng *rand.Rand, count int) []FieldElementCaseJSON {
	out := make([]FieldElementCaseJSON, count)
	for i := 0; i < count; i++ {
		a := randomBN254FPFieldBigInt(rng)
		b := randomBN254FPFieldBigInt(rng)
		if i%7 == 0 {
			b = new(big.Int).Set(a)
		}
		out[i] = buildBN254FPElementCase(fmt.Sprintf("random_%02d", i), a, b)
	}
	return out
}

func buildBLS12381FPDifferentialCases(rng *rand.Rand, count int) []FieldElementCaseJSON {
	out := make([]FieldElementCaseJSON, count)
	for i := 0; i < count; i++ {
		a := randomBLS12381FPFieldBigInt(rng)
		b := randomBLS12381FPFieldBigInt(rng)
		out[i] = buildBLS12381FPElementCase(fmt.Sprintf("random_%02d", i), a, b)
	}
	return out
}

func bn254FRToMont(regular *big.Int) gnarkbn254fr.Element {
	var z gnarkbn254fr.Element
	z.SetBigInt(regular)
	return z
}

func bls12381FRToMont(regular *big.Int) gnarkbls12381fr.Element {
	var z gnarkbls12381fr.Element
	z.SetBigInt(regular)
	return z
}

func bn254FPToMont(regular *big.Int) gnarkbn254fp.Element {
	var z gnarkbn254fp.Element
	z.SetBigInt(regular)
	return z
}

func bls12381FPToMont(regular *big.Int) gnarkbls12381fp.Element {
	var z gnarkbls12381fp.Element
	z.SetBigInt(regular)
	return z
}

func bn254FRZeroMont() gnarkbn254fr.Element {
	var z gnarkbn254fr.Element
	z.SetZero()
	return z
}

func bls12381FRZeroMont() gnarkbls12381fr.Element {
	var z gnarkbls12381fr.Element
	z.SetZero()
	return z
}

func bn254FPZeroMont() gnarkbn254fp.Element {
	var z gnarkbn254fp.Element
	z.SetZero()
	return z
}

func bls12381FPZeroMont() gnarkbls12381fp.Element {
	var z gnarkbls12381fp.Element
	z.SetZero()
	return z
}

func bn254FrElementHex(z gnarkbn254fr.Element) string {
	data := make([]byte, 32)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(data[i*8:], z[i])
	}
	return hex.EncodeToString(data)
}

func bls12381FrElementHex(z gnarkbls12381fr.Element) string {
	data := make([]byte, 32)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(data[i*8:], z[i])
	}
	return hex.EncodeToString(data)
}

func bn254FpElementHex(z gnarkbn254fp.Element) string {
	data := make([]byte, 32)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(data[i*8:], z[i])
	}
	return hex.EncodeToString(data)
}

func bls12381FpElementHex(z gnarkbls12381fp.Element) string {
	data := make([]byte, 48)
	for i := 0; i < 6; i++ {
		binary.LittleEndian.PutUint64(data[i*8:], z[i])
	}
	return hex.EncodeToString(data)
}

func bn254FRRegularHex(v *big.Int) string {
	return encodeLittleEndianBig(v, 32)
}

func blsFRRegularHex(v *big.Int) string {
	return encodeLittleEndianBig(v, 32)
}

func bn254FPRegularHex(v *big.Int) string {
	return encodeLittleEndianBig(v, 32)
}

func blsFPRegularHex(v *big.Int) string {
	return encodeLittleEndianBig(v, 48)
}

func encodeLittleEndianBig(v *big.Int, size int) string {
	bytes := v.FillBytes(make([]byte, size))
	return hex.EncodeToString(regularToLittleEndian(bytes))
}

func bn254RegularUint64(v uint64) *big.Int {
	return new(big.Int).SetUint64(v)
}

func blsRegularUint64(v uint64) *big.Int {
	return new(big.Int).SetUint64(v)
}

func bn254FRModulus() *big.Int {
	return new(big.Int).Set(gnarkbn254fr.Modulus())
}

func bn254FPModulus() *big.Int {
	return new(big.Int).Set(gnarkbn254fp.Modulus())
}

func blsFRModulus() *big.Int {
	return new(big.Int).Set(gnarkbls12381fr.Modulus())
}

func blsFPModulus() *big.Int {
	return new(big.Int).Set(gnarkbls12381fp.Modulus())
}

func bn254FRQMinusOne() *big.Int { return new(big.Int).Sub(gnarkbn254fr.Modulus(), big.NewInt(1)) }
func bn254FPQMinusOne() *big.Int { return new(big.Int).Sub(gnarkbn254fp.Modulus(), big.NewInt(1)) }
func blsFRQMinusOne() *big.Int   { return new(big.Int).Sub(gnarkbls12381fr.Modulus(), big.NewInt(1)) }
func blsFPQMinusOne() *big.Int   { return new(big.Int).Sub(gnarkbls12381fp.Modulus(), big.NewInt(1)) }

func addBig(a, b *big.Int) *big.Int { return new(big.Int).Add(a, b) }
func subBig(a, b *big.Int) *big.Int { return new(big.Int).Sub(a, b) }
func mulBig(a, b *big.Int) *big.Int { return new(big.Int).Mul(a, b) }

func bn254RegularPow2MinusOne(bitsN uint) *big.Int { return subBig(new(big.Int).Lsh(big.NewInt(1), bitsN), big.NewInt(1)) }
func blsRegularPow2MinusOne(bitsN uint) *big.Int   { return subBig(new(big.Int).Lsh(big.NewInt(1), bitsN), big.NewInt(1)) }

func bn254FRQMinus(delta uint64) *big.Int { return new(big.Int).Sub(gnarkbn254fr.Modulus(), new(big.Int).SetUint64(delta)) }
func bn254FPQMinus(delta uint64) *big.Int { return new(big.Int).Sub(gnarkbn254fp.Modulus(), new(big.Int).SetUint64(delta)) }
func blsFRQMinus(delta uint64) *big.Int   { return new(big.Int).Sub(gnarkbls12381fr.Modulus(), new(big.Int).SetUint64(delta)) }
func blsFPQMinus(delta uint64) *big.Int   { return new(big.Int).Sub(gnarkbls12381fp.Modulus(), new(big.Int).SetUint64(delta)) }

func bn254FRFloorHalfModulus() *big.Int { return new(big.Int).Rsh(bn254FRQMinusOne(), 1) }
func bn254FRCeilHalfModulus() *big.Int  { return new(big.Int).Sub(gnarkbn254fr.Modulus(), bn254FRFloorHalfModulus()) }
func bn254FPFloorHalfModulus() *big.Int { return new(big.Int).Rsh(bn254FPQMinusOne(), 1) }
func bn254FPCeilHalfModulus() *big.Int  { return new(big.Int).Sub(gnarkbn254fp.Modulus(), bn254FPFloorHalfModulus()) }
func blsFRFloorHalfModulus() *big.Int   { return new(big.Int).Rsh(blsFRQMinusOne(), 1) }
func blsFRCeilHalfModulus() *big.Int    { return new(big.Int).Sub(gnarkbls12381fr.Modulus(), blsFRFloorHalfModulus()) }
func blsFPFloorHalfModulus() *big.Int   { return new(big.Int).Rsh(blsFPQMinusOne(), 1) }
func blsFPCeilHalfModulus() *big.Int    { return new(big.Int).Sub(gnarkbls12381fp.Modulus(), blsFPFloorHalfModulus()) }

func randomBN254FRFieldBigInt(rng *rand.Rand) *big.Int {
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), gnarkbn254fr.Modulus())
}

func randomBLS12381FRFieldBigInt(rng *rand.Rand) *big.Int {
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), gnarkbls12381fr.Modulus())
}

func randomBN254FPFieldBigInt(rng *rand.Rand) *big.Int {
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), gnarkbn254fp.Modulus())
}

func randomBLS12381FPFieldBigInt(rng *rand.Rand) *big.Int {
	return new(big.Int).Rand(rng, gnarkbls12381fp.Modulus())
}
