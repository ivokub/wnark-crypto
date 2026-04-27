package testgen

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math/big"
	"math/bits"
	"math/rand"

	gnarkbls12377fp "github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	gnarkbls12377fr "github.com/consensys/gnark-crypto/ecc/bls12-377/fr"
	gnarkbls12377fft "github.com/consensys/gnark-crypto/ecc/bls12-377/fr/fft"
	"github.com/consensys/gnark-crypto/utils"
)

func BuildBLS12377FROpsVectors() FieldOpsVectorsJSON {
	rng := rand.New(rand.NewSource(20260406))
	return FieldOpsVectorsJSON{
		ElementCases: []FieldElementCaseJSON{
			buildBLS12377FRElementCase("zero_zero", bls12377RegularUint64(0), bls12377RegularUint64(0)),
			buildBLS12377FRElementCase("zero_one", bls12377RegularUint64(0), bls12377RegularUint64(1)),
			buildBLS12377FRElementCase("one_one", bls12377RegularUint64(1), bls12377RegularUint64(1)),
			buildBLS12377FRElementCase("two_five", bls12377RegularUint64(2), bls12377RegularUint64(5)),
			buildBLS12377FRElementCase("neg_one_one", bls12377FRQMinusOne(), bls12377RegularUint64(1)),
			buildBLS12377FRElementCase("seven_five", bls12377RegularUint64(7), bls12377RegularUint64(5)),
		},
		EdgeCases: []FieldElementCaseJSON{
			buildBLS12377FRElementCase("carry32_plus_one", bls12377RegularPow2MinusOne(32), bls12377RegularUint64(1)),
			buildBLS12377FRElementCase("carry64_plus_one", bls12377RegularPow2MinusOne(64), bls12377RegularUint64(1)),
			buildBLS12377FRElementCase("carry128_plus_one", bls12377RegularPow2MinusOne(128), bls12377RegularUint64(1)),
			buildBLS12377FRElementCase("carry192_plus_one", bls12377RegularPow2MinusOne(192), bls12377RegularUint64(1)),
			buildBLS12377FRElementCase("q_minus_two_plus_three", bls12377FRQMinus(2), bls12377RegularUint64(3)),
			buildBLS12377FRElementCase("q_minus_one_q_minus_one", bls12377FRQMinusOne(), bls12377FRQMinusOne()),
			buildBLS12377FRElementCase("q_minus_two_q_minus_one", bls12377FRQMinus(2), bls12377FRQMinusOne()),
			buildBLS12377FRElementCase("half_q_floor_half_q_ceil", bls12377FRFloorHalfModulus(), bls12377FRCeilHalfModulus()),
		},
		DifferentialCases: buildBLS12377FRDifferentialCases(rng, 32),
		NormalizeCases: []NormalizeCaseJSON{
			{Name: "zero", InputBytesLE: bls12377FRRegularHex(bls12377RegularUint64(0)), ExpectedBytesLE: bls12377FRRegularHex(bls12377RegularUint64(0))},
			{Name: "one", InputBytesLE: bls12377FRRegularHex(bls12377RegularUint64(1)), ExpectedBytesLE: bls12377FRRegularHex(bls12377RegularUint64(1))},
			{Name: "q_minus_one", InputBytesLE: bls12377FRRegularHex(bls12377FRQMinusOne()), ExpectedBytesLE: bls12377FRRegularHex(bls12377FRQMinusOne())},
			{Name: "q", InputBytesLE: bls12377FRRegularHex(bls12377FRModulus()), ExpectedBytesLE: bls12377FRRegularHex(bls12377RegularUint64(0))},
			{Name: "q_plus_one", InputBytesLE: bls12377FRRegularHex(addBig(bls12377FRModulus(), bls12377RegularUint64(1))), ExpectedBytesLE: bls12377FRRegularHex(bls12377RegularUint64(1))},
			{Name: "two_q_minus_one", InputBytesLE: bls12377FRRegularHex(subBig(mulBig(bls12377FRModulus(), bls12377RegularUint64(2)), bls12377RegularUint64(1))), ExpectedBytesLE: bls12377FRRegularHex(bls12377FRQMinusOne())},
		},
		ConvertCases: []ConvertCaseJSON{
			buildBLS12377FRConvertCase("zero", bls12377RegularUint64(0)),
			buildBLS12377FRConvertCase("one", bls12377RegularUint64(1)),
			buildBLS12377FRConvertCase("two", bls12377RegularUint64(2)),
			buildBLS12377FRConvertCase("five", bls12377RegularUint64(5)),
			buildBLS12377FRConvertCase("seven", bls12377RegularUint64(7)),
			buildBLS12377FRConvertCase("q_minus_one", bls12377FRQMinusOne()),
		},
	}
}

func BuildBLS12377FPOpsVectors() FieldOpsVectorsJSON {
	rng := rand.New(rand.NewSource(20260407))
	return FieldOpsVectorsJSON{
		ElementCases: []FieldElementCaseJSON{
			buildBLS12377FPElementCase("zero_zero", bls12377RegularUint64(0), bls12377RegularUint64(0)),
			buildBLS12377FPElementCase("zero_one", bls12377RegularUint64(0), bls12377RegularUint64(1)),
			buildBLS12377FPElementCase("one_one", bls12377RegularUint64(1), bls12377RegularUint64(1)),
			buildBLS12377FPElementCase("two_five", bls12377RegularUint64(2), bls12377RegularUint64(5)),
			buildBLS12377FPElementCase("neg_one_one", bls12377FPQMinusOne(), bls12377RegularUint64(1)),
			buildBLS12377FPElementCase("seven_five", bls12377RegularUint64(7), bls12377RegularUint64(5)),
		},
		EdgeCases: []FieldElementCaseJSON{
			buildBLS12377FPElementCase("carry32_plus_one", bls12377RegularPow2MinusOne(32), bls12377RegularUint64(1)),
			buildBLS12377FPElementCase("carry64_plus_one", bls12377RegularPow2MinusOne(64), bls12377RegularUint64(1)),
			buildBLS12377FPElementCase("carry128_plus_one", bls12377RegularPow2MinusOne(128), bls12377RegularUint64(1)),
			buildBLS12377FPElementCase("carry192_plus_one", bls12377RegularPow2MinusOne(192), bls12377RegularUint64(1)),
			buildBLS12377FPElementCase("q_minus_two_plus_three", bls12377FPQMinus(2), bls12377RegularUint64(3)),
			buildBLS12377FPElementCase("q_minus_one_q_minus_one", bls12377FPQMinusOne(), bls12377FPQMinusOne()),
			buildBLS12377FPElementCase("q_minus_two_q_minus_one", bls12377FPQMinus(2), bls12377FPQMinusOne()),
			buildBLS12377FPElementCase("half_q_floor_half_q_ceil", bls12377FPFloorHalfModulus(), bls12377FPCeilHalfModulus()),
		},
		DifferentialCases: buildBLS12377FPDifferentialCases(rng, 32),
		NormalizeCases: []NormalizeCaseJSON{
			{Name: "zero", InputBytesLE: bls12377FPRegularHex(bls12377RegularUint64(0)), ExpectedBytesLE: bls12377FPRegularHex(bls12377RegularUint64(0))},
			{Name: "one", InputBytesLE: bls12377FPRegularHex(bls12377RegularUint64(1)), ExpectedBytesLE: bls12377FPRegularHex(bls12377RegularUint64(1))},
			{Name: "q_minus_one", InputBytesLE: bls12377FPRegularHex(bls12377FPQMinusOne()), ExpectedBytesLE: bls12377FPRegularHex(bls12377FPQMinusOne())},
			{Name: "q", InputBytesLE: bls12377FPRegularHex(bls12377FPModulus()), ExpectedBytesLE: bls12377FPRegularHex(bls12377RegularUint64(0))},
			{Name: "q_plus_one", InputBytesLE: bls12377FPRegularHex(addBig(bls12377FPModulus(), bls12377RegularUint64(1))), ExpectedBytesLE: bls12377FPRegularHex(bls12377RegularUint64(1))},
			{Name: "two_q_minus_one", InputBytesLE: bls12377FPRegularHex(subBig(mulBig(bls12377FPModulus(), bls12377RegularUint64(2)), bls12377RegularUint64(1))), ExpectedBytesLE: bls12377FPRegularHex(bls12377FPQMinusOne())},
		},
		ConvertCases: []ConvertCaseJSON{
			buildBLS12377FPConvertCase("zero", bls12377RegularUint64(0)),
			buildBLS12377FPConvertCase("one", bls12377RegularUint64(1)),
			buildBLS12377FPConvertCase("two", bls12377RegularUint64(2)),
			buildBLS12377FPConvertCase("five", bls12377RegularUint64(5)),
			buildBLS12377FPConvertCase("seven", bls12377RegularUint64(7)),
			buildBLS12377FPConvertCase("q_minus_one", bls12377FPQMinusOne()),
		},
	}
}

func BuildBLS12377FRVectorOps() BN254VectorOpsJSON {
	return BN254VectorOpsJSON{
		VectorCases: []BN254VectorCaseJSON{
			buildBLS12377FRVectorCase("n8_random", 8, rand.New(rand.NewSource(2026040601))),
			buildBLS12377FRVectorCase("n16_random", 16, rand.New(rand.NewSource(2026040602))),
		},
	}
}

func BuildBLS12377NTTDomainFile(minLog, maxLog int) BN254NTTDomainFileJSON {
	out := BN254NTTDomainFileJSON{
		Domains: make([]BN254NTTDomainJSON, 0, maxLog-minLog+1),
	}
	for logN := minLog; logN <= maxLog; logN++ {
		size := 1 << logN
		domain := gnarkbls12377fft.NewDomain(uint64(size))
		var cosetDenInv, one gnarkbls12377fr.Element
		one.SetOne()
		cosetDenInv.Exp(domain.FrMultiplicativeGen, big.NewInt(int64(domain.Cardinality)))
		cosetDenInv.Sub(&cosetDenInv, &one).Inverse(&cosetDenInv)
		out.Domains = append(out.Domains, BN254NTTDomainJSON{
			LogN:              logN,
			Size:              size,
			OmegaHex:          domain.Generator.BigInt(new(big.Int)).Text(16),
			OmegaInvHex:       domain.GeneratorInv.BigInt(new(big.Int)).Text(16),
			CardinalityInvHex: domain.CardinalityInv.BigInt(new(big.Int)).Text(16),
			CosetGenHex:       domain.FrMultiplicativeGen.BigInt(new(big.Int)).Text(16),
			CosetGenInvHex:    domain.FrMultiplicativeGenInv.BigInt(new(big.Int)).Text(16),
			CosetDenInvHex:    cosetDenInv.BigInt(new(big.Int)).Text(16),
		})
	}
	return out
}

func BuildBLS12377NTTVectors() BN254NTTVectorsJSON {
	return BN254NTTVectorsJSON{
		NTTCases: []BN254NTTCaseJSON{
			buildBLS12377NTTCase("n8_random", 8, rand.New(rand.NewSource(2026040603))),
			buildBLS12377NTTCase("n16_random", 16, rand.New(rand.NewSource(2026040604))),
		},
	}
}

func buildBLS12377FRVectorCase(name string, size int, rng *rand.Rand) BN254VectorCaseJSON {
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
		aRegular := randomBLS12377FRFieldBigInt(rng)
		bRegular := randomBLS12377FRFieldBigInt(rng)
		var aMont, bMont gnarkbls12377fr.Element
		aMont.SetBigInt(aRegular)
		bMont.SetBigInt(bRegular)

		var add, sub, mul gnarkbls12377fr.Element
		add.Add(&aMont, &bMont)
		sub.Sub(&aMont, &bMont)
		mul.Mul(&aMont, &bMont)

		out.RegularInputs[i] = bls12377FRRegularHex(aRegular)
		out.MontInputs[i] = bls12377FrElementHex(aMont)
		out.MontFactors[i] = bls12377FrElementHex(bMont)
		out.AddExpected[i] = bls12377FrElementHex(add)
		out.SubExpected[i] = bls12377FrElementHex(sub)
		out.MulExpected[i] = bls12377FrElementHex(mul)
		out.ToMontExpected[i] = bls12377FrElementHex(aMont)
		out.FromMontExpected[i] = bls12377FRRegularHex(aRegular)
	}

	logCount := bits.Len(uint(size)) - 1
	for i := 0; i < size; i++ {
		j := int(bits.Reverse64(uint64(i)) >> (64 - logCount))
		out.BitReverseExpected[i] = out.MontInputs[j]
	}

	return out
}

func buildBLS12377NTTCase(name string, size int, rng *rand.Rand) BN254NTTCaseJSON {
	domain := gnarkbls12377fft.NewDomain(uint64(size))
	twiddles, err := domain.Twiddles()
	if err != nil {
		panic(err)
	}
	twiddlesInv, err := domain.TwiddlesInv()
	if err != nil {
		panic(err)
	}

	input := make([]gnarkbls12377fr.Element, size)
	for i := range input {
		input[i].SetBigInt(randomBLS12377FRFieldBigInt(rng))
	}

	forward := make([]gnarkbls12377fr.Element, size)
	copy(forward, input)
	utils.BitReverse(forward)
	domain.FFT(forward, gnarkbls12377fft.DIT)

	inverse := make([]gnarkbls12377fr.Element, size)
	copy(inverse, forward)
	utils.BitReverse(inverse)
	domain.FFTInverse(inverse, gnarkbls12377fft.DIT)

	logN := bits.Len(uint(size)) - 1
	stageTwiddles := make([][]string, logN)
	inverseStageTwiddles := make([][]string, logN)
	for stage := 1; stage <= logN; stage++ {
		m := 1 << (stage - 1)
		src := twiddles[logN-stage]
		srcInv := twiddlesInv[logN-stage]
		stageTwiddles[stage-1] = make([]string, m)
		inverseStageTwiddles[stage-1] = make([]string, m)
		for i := 0; i < m; i++ {
			stageTwiddles[stage-1][i] = bls12377FrElementHex(src[i])
			inverseStageTwiddles[stage-1][i] = bls12377FrElementHex(srcInv[i])
		}
	}

	return BN254NTTCaseJSON{
		Name:                   name,
		Size:                   size,
		InputMontLE:            encodeBLS12377FRBatch(input),
		ForwardExpectedLE:      encodeBLS12377FRBatch(forward),
		InverseExpectedLE:      encodeBLS12377FRBatch(inverse),
		StageTwiddlesLE:        stageTwiddles,
		InverseStageTwiddlesLE: inverseStageTwiddles,
		InverseScaleLE:         bls12377FrElementHex(domain.CardinalityInv),
	}
}

func buildBLS12377FRElementCase(name string, aRegular, bRegular *big.Int) FieldElementCaseJSON {
	aMont := bls12377FRToMont(aRegular)
	bMont := bls12377FRToMont(bRegular)
	var add, sub, negA, dblA, mul, sqA gnarkbls12377fr.Element
	add.Add(&aMont, &bMont)
	sub.Sub(&aMont, &bMont)
	negA.Neg(&aMont)
	dblA.Double(&aMont)
	mul.Mul(&aMont, &bMont)
	sqA.Square(&aMont)
	equal := bls12377FRZeroMont()
	if aMont.Equal(&bMont) {
		equal.SetUint64(1)
	}
	return FieldElementCaseJSON{
		Name:           name,
		ABytesLE:       bls12377FrElementHex(aMont),
		BBytesLE:       bls12377FrElementHex(bMont),
		EqualBytesLE:   bls12377FrElementHex(equal),
		AddBytesLE:     bls12377FrElementHex(add),
		SubBytesLE:     bls12377FrElementHex(sub),
		NegABytesLE:    bls12377FrElementHex(negA),
		DoubleABytesLE: bls12377FrElementHex(dblA),
		MulBytesLE:     bls12377FrElementHex(mul),
		SquareABytesLE: bls12377FrElementHex(sqA),
	}
}

func buildBLS12377FPElementCase(name string, aRegular, bRegular *big.Int) FieldElementCaseJSON {
	aMont := bls12377FPToMont(aRegular)
	bMont := bls12377FPToMont(bRegular)
	var add, sub, negA, dblA, mul, sqA gnarkbls12377fp.Element
	add.Add(&aMont, &bMont)
	sub.Sub(&aMont, &bMont)
	negA.Neg(&aMont)
	dblA.Double(&aMont)
	mul.Mul(&aMont, &bMont)
	sqA.Square(&aMont)
	equal := bls12377FPZeroMont()
	if aMont.Equal(&bMont) {
		equal.SetUint64(1)
	}
	return FieldElementCaseJSON{
		Name:           name,
		ABytesLE:       bls12377FpElementHex(aMont),
		BBytesLE:       bls12377FpElementHex(bMont),
		EqualBytesLE:   bls12377FpElementHex(equal),
		AddBytesLE:     bls12377FpElementHex(add),
		SubBytesLE:     bls12377FpElementHex(sub),
		NegABytesLE:    bls12377FpElementHex(negA),
		DoubleABytesLE: bls12377FpElementHex(dblA),
		MulBytesLE:     bls12377FpElementHex(mul),
		SquareABytesLE: bls12377FpElementHex(sqA),
	}
}

func buildBLS12377FRConvertCase(name string, regular *big.Int) ConvertCaseJSON {
	return ConvertCaseJSON{Name: name, RegularBytes: bls12377FRRegularHex(regular), MontBytes: bls12377FrElementHex(bls12377FRToMont(regular))}
}

func buildBLS12377FPConvertCase(name string, regular *big.Int) ConvertCaseJSON {
	return ConvertCaseJSON{Name: name, RegularBytes: bls12377FPRegularHex(regular), MontBytes: bls12377FpElementHex(bls12377FPToMont(regular))}
}

func buildBLS12377FRDifferentialCases(rng *rand.Rand, count int) []FieldElementCaseJSON {
	out := make([]FieldElementCaseJSON, count)
	for i := 0; i < count; i++ {
		a := randomBLS12377FRFieldBigInt(rng)
		b := randomBLS12377FRFieldBigInt(rng)
		if i%7 == 0 {
			b = new(big.Int).Set(a)
		}
		out[i] = buildBLS12377FRElementCase(fmt.Sprintf("random_%02d", i), a, b)
	}
	return out
}

func buildBLS12377FPDifferentialCases(rng *rand.Rand, count int) []FieldElementCaseJSON {
	out := make([]FieldElementCaseJSON, count)
	for i := 0; i < count; i++ {
		a := randomBLS12377FPFieldBigInt(rng)
		b := randomBLS12377FPFieldBigInt(rng)
		out[i] = buildBLS12377FPElementCase(fmt.Sprintf("random_%02d", i), a, b)
	}
	return out
}

func encodeBLS12377FRBatch(in []gnarkbls12377fr.Element) []string {
	out := make([]string, len(in))
	for i := range in {
		out[i] = bls12377FrElementHex(in[i])
	}
	return out
}

func bls12377FRToMont(regular *big.Int) gnarkbls12377fr.Element {
	var z gnarkbls12377fr.Element
	z.SetBigInt(regular)
	return z
}

func bls12377FPToMont(regular *big.Int) gnarkbls12377fp.Element {
	var z gnarkbls12377fp.Element
	z.SetBigInt(regular)
	return z
}

func bls12377FRZeroMont() gnarkbls12377fr.Element {
	var z gnarkbls12377fr.Element
	z.SetZero()
	return z
}

func bls12377FPZeroMont() gnarkbls12377fp.Element {
	var z gnarkbls12377fp.Element
	z.SetZero()
	return z
}

func bls12377FrElementHex(z gnarkbls12377fr.Element) string {
	data := make([]byte, 32)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(data[i*8:], z[i])
	}
	return hex.EncodeToString(data)
}

func bls12377FpElementHex(z gnarkbls12377fp.Element) string {
	data := make([]byte, 48)
	for i := 0; i < 6; i++ {
		binary.LittleEndian.PutUint64(data[i*8:], z[i])
	}
	return hex.EncodeToString(data)
}

func bls12377FRRegularHex(v *big.Int) string {
	return encodeLittleEndianBig(v, 32)
}

func bls12377FPRegularHex(v *big.Int) string {
	return encodeLittleEndianBig(v, 48)
}

func bls12377RegularUint64(v uint64) *big.Int {
	return new(big.Int).SetUint64(v)
}

func bls12377FRModulus() *big.Int {
	return new(big.Int).Set(gnarkbls12377fr.Modulus())
}

func bls12377FPModulus() *big.Int {
	return new(big.Int).Set(gnarkbls12377fp.Modulus())
}

func bls12377FRQMinusOne() *big.Int {
	return new(big.Int).Sub(gnarkbls12377fr.Modulus(), big.NewInt(1))
}
func bls12377FPQMinusOne() *big.Int {
	return new(big.Int).Sub(gnarkbls12377fp.Modulus(), big.NewInt(1))
}

func bls12377RegularPow2MinusOne(bitsN uint) *big.Int {
	return subBig(new(big.Int).Lsh(big.NewInt(1), bitsN), big.NewInt(1))
}

func bls12377FRQMinus(delta uint64) *big.Int {
	return new(big.Int).Sub(gnarkbls12377fr.Modulus(), new(big.Int).SetUint64(delta))
}

func bls12377FPQMinus(delta uint64) *big.Int {
	return new(big.Int).Sub(gnarkbls12377fp.Modulus(), new(big.Int).SetUint64(delta))
}

func bls12377FRFloorHalfModulus() *big.Int { return new(big.Int).Rsh(bls12377FRQMinusOne(), 1) }
func bls12377FRCeilHalfModulus() *big.Int {
	return new(big.Int).Sub(gnarkbls12377fr.Modulus(), bls12377FRFloorHalfModulus())
}
func bls12377FPFloorHalfModulus() *big.Int { return new(big.Int).Rsh(bls12377FPQMinusOne(), 1) }
func bls12377FPCeilHalfModulus() *big.Int {
	return new(big.Int).Sub(gnarkbls12377fp.Modulus(), bls12377FPFloorHalfModulus())
}

func randomBLS12377FRFieldBigInt(rng *rand.Rand) *big.Int {
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), gnarkbls12377fr.Modulus())
}

func randomBLS12377FPFieldBigInt(rng *rand.Rand) *big.Int {
	return new(big.Int).Rand(rng, gnarkbls12377fp.Modulus())
}
