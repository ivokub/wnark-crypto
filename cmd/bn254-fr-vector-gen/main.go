package main

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"math/rand"
	"os"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

type phase2Vectors struct {
	ElementCases      []elementCase   `json:"element_cases"`
	EdgeCases         []elementCase   `json:"edge_cases"`
	DifferentialCases []elementCase   `json:"differential_cases"`
	NormalizeCases    []normalizeCase `json:"normalize_cases"`
	ConvertCases      []convertCase   `json:"convert_cases"`
}

type elementCase struct {
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

type normalizeCase struct {
	Name            string `json:"name"`
	InputBytesLE    string `json:"input_bytes_le"`
	ExpectedBytesLE string `json:"expected_bytes_le"`
}

type convertCase struct {
	Name         string `json:"name"`
	RegularBytes string `json:"regular_bytes_le"`
	MontBytes    string `json:"mont_bytes_le"`
}

func main() {
	outPath := flag.String("out", "", "write JSON output to path")
	flag.Parse()

	rng := rand.New(rand.NewSource(20260402))

	vectors := phase2Vectors{
		ElementCases: []elementCase{
			buildElementCase("zero_zero", regularUint64(0), regularUint64(0)),
			buildElementCase("zero_one", regularUint64(0), regularUint64(1)),
			buildElementCase("one_one", regularUint64(1), regularUint64(1)),
			buildElementCase("two_five", regularUint64(2), regularUint64(5)),
			buildElementCase("neg_one_one", regularQMinusOne(), regularUint64(1)),
			buildElementCase("seven_five", regularUint64(7), regularUint64(5)),
		},
		EdgeCases: []elementCase{
			buildElementCase("carry32_plus_one", regularPow2MinusOne(32), regularUint64(1)),
			buildElementCase("carry64_plus_one", regularPow2MinusOne(64), regularUint64(1)),
			buildElementCase("carry128_plus_one", regularPow2MinusOne(128), regularUint64(1)),
			buildElementCase("carry192_plus_one", regularPow2MinusOne(192), regularUint64(1)),
			buildElementCase("q_minus_two_plus_three", regularQMinus(2), regularUint64(3)),
			buildElementCase("q_minus_one_q_minus_one", regularQMinusOne(), regularQMinusOne()),
			buildElementCase("q_minus_two_q_minus_one", regularQMinus(2), regularQMinusOne()),
			buildElementCase("half_q_floor_half_q_ceil", regularFloorHalfModulus(), regularCeilHalfModulus()),
		},
		DifferentialCases: buildDifferentialCases(rng, 32),
		NormalizeCases: []normalizeCase{
			{
				Name:            "zero",
				InputBytesLE:    regularHex(regularUint64(0)),
				ExpectedBytesLE: regularHex(regularUint64(0)),
			},
			{
				Name:            "one",
				InputBytesLE:    regularHex(regularUint64(1)),
				ExpectedBytesLE: regularHex(regularUint64(1)),
			},
			{
				Name:            "q_minus_one",
				InputBytesLE:    regularHex(regularQMinusOne()),
				ExpectedBytesLE: regularHex(regularQMinusOne()),
			},
			{
				Name:            "q",
				InputBytesLE:    regularHex(regularModulus()),
				ExpectedBytesLE: regularHex(regularUint64(0)),
			},
			{
				Name:            "q_plus_one",
				InputBytesLE:    regularHex(addBig(regularModulus(), regularUint64(1))),
				ExpectedBytesLE: regularHex(regularUint64(1)),
			},
			{
				Name:            "two_q_minus_one",
				InputBytesLE:    regularHex(subBig(mulBig(regularModulus(), regularUint64(2)), regularUint64(1))),
				ExpectedBytesLE: regularHex(regularQMinusOne()),
			},
		},
		ConvertCases: []convertCase{
			buildConvertCase("zero", regularUint64(0)),
			buildConvertCase("one", regularUint64(1)),
			buildConvertCase("two", regularUint64(2)),
			buildConvertCase("five", regularUint64(5)),
			buildConvertCase("seven", regularUint64(7)),
			buildConvertCase("q_minus_one", regularQMinusOne()),
		},
	}

	data, err := json.MarshalIndent(vectors, "", "  ")
	if err != nil {
		panic(err)
	}
	data = append(data, '\n')

	if *outPath == "" {
		if _, err := os.Stdout.Write(data); err != nil {
			panic(err)
		}
		return
	}
	if err := os.WriteFile(*outPath, data, 0o644); err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", *outPath)
}

func buildElementCase(name string, aRegular, bRegular *big.Int) elementCase {
	aMont := toMont(aRegular)
	bMont := toMont(bRegular)

	var add, sub, negA, dblA, mul, sqA gnarkfr.Element
	add.Add(&aMont, &bMont)
	sub.Sub(&aMont, &bMont)
	negA.Neg(&aMont)
	dblA.Double(&aMont)
	mul.Mul(&aMont, &bMont)
	sqA.Square(&aMont)

	equal := zeroMont()
	if aMont.Equal(&bMont) {
		equal.SetUint64(1)
	}

	return elementCase{
		Name:           name,
		ABytesLE:       montHex(aMont),
		BBytesLE:       montHex(bMont),
		EqualBytesLE:   montHex(equal),
		AddBytesLE:     montHex(add),
		SubBytesLE:     montHex(sub),
		NegABytesLE:    montHex(negA),
		DoubleABytesLE: montHex(dblA),
		MulBytesLE:     montHex(mul),
		SquareABytesLE: montHex(sqA),
	}
}

func buildConvertCase(name string, regular *big.Int) convertCase {
	return convertCase{
		Name:         name,
		RegularBytes: regularHex(regular),
		MontBytes:    montHex(toMont(regular)),
	}
}

func toMont(regular *big.Int) gnarkfr.Element {
	var z gnarkfr.Element
	z.SetBigInt(regular)
	return z
}

func zeroMont() gnarkfr.Element {
	var z gnarkfr.Element
	z.SetZero()
	return z
}

func montHex(z gnarkfr.Element) string {
	data := make([]byte, 32)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(data[i*8:], z[i])
	}
	return hex.EncodeToString(data)
}

func regularHex(v *big.Int) string {
	bytes := v.FillBytes(make([]byte, 32))
	for i, j := 0, len(bytes)-1; i < j; i, j = i+1, j-1 {
		bytes[i], bytes[j] = bytes[j], bytes[i]
	}
	return hex.EncodeToString(bytes)
}

func regularUint64(v uint64) *big.Int {
	return new(big.Int).SetUint64(v)
}

func regularModulus() *big.Int {
	return new(big.Int).Set(gnarkfr.Modulus())
}

func regularQMinusOne() *big.Int {
	return new(big.Int).Sub(gnarkfr.Modulus(), big.NewInt(1))
}

func addBig(a, b *big.Int) *big.Int {
	return new(big.Int).Add(a, b)
}

func subBig(a, b *big.Int) *big.Int {
	return new(big.Int).Sub(a, b)
}

func mulBig(a, b *big.Int) *big.Int {
	return new(big.Int).Mul(a, b)
}

func regularPow2MinusOne(bits uint) *big.Int {
	one := big.NewInt(1)
	return new(big.Int).Sub(new(big.Int).Lsh(one, bits), one)
}

func regularQMinus(delta uint64) *big.Int {
	return new(big.Int).Sub(gnarkfr.Modulus(), new(big.Int).SetUint64(delta))
}

func regularFloorHalfModulus() *big.Int {
	return new(big.Int).Rsh(regularQMinusOne(), 1)
}

func regularCeilHalfModulus() *big.Int {
	return new(big.Int).Sub(gnarkfr.Modulus(), regularFloorHalfModulus())
}

func buildDifferentialCases(rng *rand.Rand, count int) []elementCase {
	out := make([]elementCase, count)
	for i := 0; i < count; i++ {
		a := randomFieldBigInt(rng)
		b := randomFieldBigInt(rng)
		if i%7 == 0 {
			b = new(big.Int).Set(a)
		}
		out[i] = buildElementCase(fmt.Sprintf("random_%02d", i), a, b)
	}
	return out
}

func randomFieldBigInt(rng *rand.Rand) *big.Int {
	buf := make([]byte, 48)
	for i := range buf {
		buf[i] = byte(rng.Uint32())
	}
	return new(big.Int).Mod(new(big.Int).SetBytes(buf), gnarkfr.Modulus())
}
