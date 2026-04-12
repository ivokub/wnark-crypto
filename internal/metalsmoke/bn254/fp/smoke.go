package smoke

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"runtime"

	gnarkfp "github.com/consensys/gnark-crypto/ecc/bn254/fp"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bn254gpu "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type fpOpsVectors struct {
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

func Run() error { return run() }

func run() error {
	fmt.Println("=== BN254 fp Ops Metal Smoke ===")
	fmt.Println()

	vectors, err := loadVectors()
	if err != nil {
		return err
	}

	deviceSet, err := bn254gpu.NewHeadlessDevice()
	if err != nil {
		return err
	}
	defer deviceSet.Close()

	kernel, err := bn254gpu.NewFpKernel(deviceSet.Device)
	if err != nil {
		return err
	}
	defer kernel.Close()

	fmt.Printf("1. Adapter: %s\n", deviceSet.Adapter.Info().Name)
	fmt.Printf("2. Sanity element cases: %d\n", len(vectors.ElementCases))
	fmt.Printf("3. Edge element cases: %d\n", len(vectors.EdgeCases))
	fmt.Printf("4. Differential element cases: %d\n", len(vectors.DifferentialCases))
	fmt.Printf("5. Normalize cases: %d\n", len(vectors.NormalizeCases))
	fmt.Printf("6. Convert cases: %d\n", len(vectors.ConvertCases))

	allElementCases := combineElementCases(vectors.ElementCases, vectors.EdgeCases, vectors.DifferentialCases)

	if err := verifyStaticOps(kernel, allElementCases, vectors.ConvertCases); err != nil {
		return err
	}
	if err := verifyBinaryOp(kernel, "add", bn254gpu.FpOpAdd, allElementCases, func(tc elementCase) string { return tc.AddBytesLE }, applyAdd); err != nil {
		return err
	}
	if err := verifyBinaryOp(kernel, "sub", bn254gpu.FpOpSub, allElementCases, func(tc elementCase) string { return tc.SubBytesLE }, applySub); err != nil {
		return err
	}
	if err := verifyUnaryOp(kernel, "neg", bn254gpu.FpOpNeg, allElementCases, func(tc elementCase) string { return tc.NegABytesLE }, applyNeg); err != nil {
		return err
	}
	if err := verifyUnaryOp(kernel, "double", bn254gpu.FpOpDouble, allElementCases, func(tc elementCase) string { return tc.DoubleABytesLE }, applyDouble); err != nil {
		return err
	}
	if err := verifyBinaryOp(kernel, "mul", bn254gpu.FpOpMul, allElementCases, func(tc elementCase) string { return tc.MulBytesLE }, applyMul); err != nil {
		return err
	}
	if err := verifyUnaryOp(kernel, "square", bn254gpu.FpOpSquare, allElementCases, func(tc elementCase) string { return tc.SquareABytesLE }, applySquare); err != nil {
		return err
	}
	if err := verifyToMont(kernel, vectors.ConvertCases); err != nil {
		return err
	}
	if err := verifyFromMont(kernel, vectors.ConvertCases); err != nil {
		return err
	}
	if err := verifyNormalize(kernel, vectors.NormalizeCases); err != nil {
		return err
	}

	fmt.Println()
	fmt.Println("PASS: BN254 fp Ops Metal smoke succeeded")
	return nil
}

func verifyStaticOps(kernel *bn254gpu.FpKernel, cases []elementCase, convertCases []convertCase) error {
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := make([]curvegpu.U32x8, len(cases))
	copyExpected := make([]string, len(cases))
	equalExpected := make([]string, len(cases))
	for i, tc := range cases {
		aBatch[i] = mustU32x8(tc.ABytesLE)
		bBatch[i] = mustU32x8(tc.BBytesLE)
		copyExpected[i] = tc.ABytesLE
		equalExpected[i] = tc.EqualBytesLE
	}
	if err := verifyBatch(kernel, "copy", bn254gpu.FpOpCopy, aBatch, bBatch, copyExpected); err != nil {
		return err
	}
	if err := verifyBatch(kernel, "equal", bn254gpu.FpOpEqual, aBatch, bBatch, equalExpected); err != nil {
		return err
	}

	zeros := bn254gpu.ZeroBatch(len(cases))
	zeroExpected := make([]string, len(cases))
	oneExpected := make([]string, len(cases))
	oneMont := mustFindConvertCase(convertCases, "one").MontBytes
	for i := range cases {
		zeroExpected[i] = "0000000000000000000000000000000000000000000000000000000000000000"
		oneExpected[i] = oneMont
	}
	if err := verifyBatch(kernel, "zero", bn254gpu.FpOpZero, zeros, zeros, zeroExpected); err != nil {
		return err
	}
	if err := verifyBatch(kernel, "one", bn254gpu.FpOpOne, zeros, zeros, oneExpected); err != nil {
		return err
	}
	return nil
}

func verifyBinaryOp(kernel *bn254gpu.FpKernel, name string, op bn254gpu.FpOp, cases []elementCase, expected func(elementCase) string, cpu func(a, b gnarkfp.Element) gnarkfp.Element) error {
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := make([]curvegpu.U32x8, len(cases))
	expectedBatch := make([]string, len(cases))
	for i, tc := range cases {
		aBatch[i] = mustU32x8(tc.ABytesLE)
		bBatch[i] = mustU32x8(tc.BBytesLE)
		expectedBatch[i] = expected(tc)

		aCPU := bytesToElement(tc.ABytesLE)
		bCPU := bytesToElement(tc.BBytesLE)
		gotCPU := elementToHex(cpu(aCPU, bCPU))
		if gotCPU != expectedBatch[i] {
			return fmt.Errorf("%s cpu mismatch for %s: got=%s want=%s", name, tc.Name, gotCPU, expectedBatch[i])
		}
	}
	return verifyBatch(kernel, name, op, aBatch, bBatch, expectedBatch)
}

func verifyUnaryOp(kernel *bn254gpu.FpKernel, name string, op bn254gpu.FpOp, cases []elementCase, expected func(elementCase) string, cpu func(a gnarkfp.Element) gnarkfp.Element) error {
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := bn254gpu.ZeroBatch(len(cases))
	expectedBatch := make([]string, len(cases))
	for i, tc := range cases {
		aBatch[i] = mustU32x8(tc.ABytesLE)
		expectedBatch[i] = expected(tc)

		aCPU := bytesToElement(tc.ABytesLE)
		gotCPU := elementToHex(cpu(aCPU))
		if gotCPU != expectedBatch[i] {
			return fmt.Errorf("%s cpu mismatch for %s: got=%s want=%s", name, tc.Name, gotCPU, expectedBatch[i])
		}
	}
	return verifyBatch(kernel, name, op, aBatch, bBatch, expectedBatch)
}

func verifyNormalize(kernel *bn254gpu.FpKernel, cases []normalizeCase) error {
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := bn254gpu.ZeroBatch(len(cases))
	expectedBatch := make([]string, len(cases))
	for i, tc := range cases {
		aBatch[i] = mustU32x8(tc.InputBytesLE)
		expectedBatch[i] = tc.ExpectedBytesLE
	}
	return verifyBatch(kernel, "normalize", bn254gpu.FpOpNormalize, aBatch, bBatch, expectedBatch)
}

func verifyToMont(kernel *bn254gpu.FpKernel, cases []convertCase) error {
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := bn254gpu.ZeroBatch(len(cases))
	expectedBatch := make([]string, len(cases))
	for i, tc := range cases {
		aBatch[i] = mustU32x8(tc.RegularBytes)
		expectedBatch[i] = tc.MontBytes
		gotCPU := elementToHex(regularBytesToMontElement(tc.RegularBytes))
		if gotCPU != expectedBatch[i] {
			return fmt.Errorf("to_mont cpu mismatch for %s: got=%s want=%s", tc.Name, gotCPU, expectedBatch[i])
		}
	}
	return verifyBatch(kernel, "to_mont", bn254gpu.FpOpToMont, aBatch, bBatch, expectedBatch)
}

func verifyFromMont(kernel *bn254gpu.FpKernel, cases []convertCase) error {
	aBatch := make([]curvegpu.U32x8, len(cases))
	bBatch := bn254gpu.ZeroBatch(len(cases))
	expectedBatch := make([]string, len(cases))
	for i, tc := range cases {
		aBatch[i] = mustU32x8(tc.MontBytes)
		expectedBatch[i] = tc.RegularBytes
		gotCPU := elementToRegularHex(bytesToElement(tc.MontBytes))
		if gotCPU != expectedBatch[i] {
			return fmt.Errorf("from_mont cpu mismatch for %s: got=%s want=%s", tc.Name, gotCPU, expectedBatch[i])
		}
	}
	return verifyBatch(kernel, "from_mont", bn254gpu.FpOpFromMont, aBatch, bBatch, expectedBatch)
}

func verifyBatch(kernel *bn254gpu.FpKernel, name string, op bn254gpu.FpOp, aBatch, bBatch []curvegpu.U32x8, expected []string) error {
	fmt.Printf("4. %s... ", name)
	got, err := kernel.Run(op, aBatch, bBatch)
	if err != nil {
		return fmt.Errorf("%s gpu run: %w", name, err)
	}
	for i, out := range got {
		gotHex := limbsToHex(out)
		if gotHex != expected[i] {
			return fmt.Errorf("%s mismatch at index %d: got=%s want=%s", name, i, gotHex, expected[i])
		}
	}
	fmt.Println("OK")
	return nil
}

func applyAdd(a, b gnarkfp.Element) gnarkfp.Element {
	var z gnarkfp.Element
	z.Add(&a, &b)
	return z
}

func applySub(a, b gnarkfp.Element) gnarkfp.Element {
	var z gnarkfp.Element
	z.Sub(&a, &b)
	return z
}

func applyNeg(a gnarkfp.Element) gnarkfp.Element {
	var z gnarkfp.Element
	z.Neg(&a)
	return z
}

func applyDouble(a gnarkfp.Element) gnarkfp.Element {
	var z gnarkfp.Element
	z.Double(&a)
	return z
}

func applyMul(a, b gnarkfp.Element) gnarkfp.Element {
	var z gnarkfp.Element
	z.Mul(&a, &b)
	return z
}

func applySquare(a gnarkfp.Element) gnarkfp.Element {
	var z gnarkfp.Element
	z.Square(&a)
	return z
}

func loadVectors() (fpOpsVectors, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return fpOpsVectors{}, fmt.Errorf("runtime caller lookup failed")
	}
	path := filepath.Join(filepath.Dir(file), "..", "..", "..", "..", "testdata", "vectors", "fp", "bn254_fp_ops.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return fpOpsVectors{}, fmt.Errorf("read vectors: %w", err)
	}
	var vectors fpOpsVectors
	if err := json.Unmarshal(data, &vectors); err != nil {
		return fpOpsVectors{}, fmt.Errorf("unmarshal vectors: %w", err)
	}
	return vectors, nil
}

func combineElementCases(groups ...[]elementCase) []elementCase {
	var total int
	for _, group := range groups {
		total += len(group)
	}
	out := make([]elementCase, 0, total)
	for _, group := range groups {
		out = append(out, group...)
	}
	return out
}

func mustU32x8(raw string) curvegpu.U32x8 {
	var out curvegpu.U32x8
	data, err := hex.DecodeString(raw)
	if err != nil {
		panic(err)
	}
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(data[i*4:])
	}
	return out
}

func limbsToHex(in curvegpu.U32x8) string {
	data := make([]byte, 32)
	for i, limb := range in {
		binary.LittleEndian.PutUint32(data[i*4:], limb)
	}
	return hex.EncodeToString(data)
}

func bytesToElement(raw string) gnarkfp.Element {
	data, err := hex.DecodeString(raw)
	if err != nil {
		panic(err)
	}
	var words [4]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(data[i*8:])
	}
	return gnarkfp.Element(words)
}

func elementToHex(in gnarkfp.Element) string {
	data := make([]byte, 32)
	for i := range in {
		binary.LittleEndian.PutUint64(data[i*8:], in[i])
	}
	return hex.EncodeToString(data)
}

func elementToRegularHex(in gnarkfp.Element) string {
	regular := in.BigInt(new(big.Int))
	bytes := regular.FillBytes(make([]byte, 32))
	for i, j := 0, len(bytes)-1; i < j; i, j = i+1, j-1 {
		bytes[i], bytes[j] = bytes[j], bytes[i]
	}
	return hex.EncodeToString(bytes)
}

func regularBytesToMontElement(raw string) gnarkfp.Element {
	data, err := hex.DecodeString(raw)
	if err != nil {
		panic(err)
	}
	for i, j := 0, len(data)-1; i < j; i, j = i+1, j-1 {
		data[i], data[j] = data[j], data[i]
	}
	var z gnarkfp.Element
	z.SetBigInt(new(big.Int).SetBytes(data))
	return z
}

func mustFindConvertCase(cases []convertCase, name string) convertCase {
	for _, tc := range cases {
		if tc.Name == name {
			return tc
		}
	}
	panic("missing convert case: " + name)
}
