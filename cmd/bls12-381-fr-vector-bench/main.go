package main

import (
	"flag"
	"fmt"
	"math/bits"
	mrand "math/rand"
	"strings"
	"time"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bls12381 "github.com/ivokub/wnark-crypto/go/curvegpu/bls12_381"
)

type benchResult struct {
	name   string
	cpu    time.Duration
	cold   bls12381.FrVectorProfile
	warm   bls12381.FrVectorProfile
	verify bool
}

func main() {
	var (
		minLog = flag.Int("min-log", 10, "minimum input size log2")
		maxLog = flag.Int("max-log", 20, "maximum input size log2")
		iters  = flag.Int("iters", 3, "iterations per benchmark")
	)
	flag.Parse()

	if *minLog < 1 || *maxLog < *minLog {
		panic("invalid min-log/max-log")
	}
	if *iters <= 0 {
		panic("iters must be positive")
	}

	fmt.Println("=== BLS12-381 fr Vector Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", *minLog, *maxLog)
	fmt.Printf("iterations: %d\n\n", *iters)

	fmt.Println("size,op,init_ms,cpu_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_total_ms,cold_with_init_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_total_ms,verified")
	for logSize := *minLog; logSize <= *maxLog; logSize++ {
		size := 1 << logSize
		left, right := makeInputs(size)
		leftGPU := vectorToGPU(left)
		rightGPU := vectorToGPU(right)

		initStart := time.Now()
		dev, err := bls12381.NewHeadlessDevice()
		if err != nil {
			panic(err)
		}
		vectorKernel, err := bls12381.NewFrVectorKernel(dev.Device)
		if err != nil {
			dev.Close()
			panic(err)
		}
		initElapsed := time.Since(initStart)

		results, err := runBenchmarks(vectorKernel, left, right, leftGPU, rightGPU, *iters)
		vectorKernel.Close()
		dev.Close()
		if err != nil {
			if isResourceError(err) {
				fmt.Printf("# stop at size=2^%d: %v\n", logSize, err)
				break
			}
			panic(err)
		}
		for _, result := range results {
			fmt.Printf(
				"%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%t\n",
				size,
				result.name,
				durationMS(initElapsed),
				durationMS(result.cpu),
				durationMS(result.cold.Upload),
				durationMS(result.cold.Kernel),
				durationMS(result.cold.Readback),
				durationMS(result.cold.Total),
				durationMS(initElapsed+result.cold.Total),
				durationMS(result.warm.Upload),
				durationMS(result.warm.Kernel),
				durationMS(result.warm.Readback),
				durationMS(result.warm.Total),
				result.verify,
			)
		}
	}
}

func runBenchmarks(
	vectorKernel *bls12381.FrVectorKernel,
	left, right gnarkfr.Vector,
	leftGPU, rightGPU []curvegpu.U32x8,
	iters int,
) ([]benchResult, error) {
	addExpected := make(gnarkfr.Vector, len(left))
	addExpected.Add(left, right)
	subExpected := make(gnarkfr.Vector, len(left))
	subExpected.Sub(left, right)
	mulExpected := make(gnarkfr.Vector, len(left))
	mulExpected.Mul(left, right)
	bitReverseExpected := bitReverse(left)

	results := make([]benchResult, 0, 4)

	cpuAdd := timeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Add(left, right)
	})
	coldAdd, warmAdd, gpuAddOut, err := timeGPUProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpAdd, leftGPU, rightGPU, 0)
	})
	if err != nil {
		return nil, fmt.Errorf("bench add: %w", err)
	}
	results = append(results, benchResult{"add", cpuAdd, coldAdd, warmAdd, equalGPUBatches(gpuAddOut, vectorToGPU(addExpected))})

	cpuSub := timeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Sub(left, right)
	})
	coldSub, warmSub, gpuSubOut, err := timeGPUProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpSub, leftGPU, rightGPU, 0)
	})
	if err != nil {
		return nil, fmt.Errorf("bench sub: %w", err)
	}
	results = append(results, benchResult{"sub", cpuSub, coldSub, warmSub, equalGPUBatches(gpuSubOut, vectorToGPU(subExpected))})

	cpuMul := timeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Mul(left, right)
	})
	coldMul, warmMul, gpuMulOut, err := timeGPUProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpMulFactors, leftGPU, rightGPU, 0)
	})
	if err != nil {
		return nil, fmt.Errorf("bench mul: %w", err)
	}
	results = append(results, benchResult{"mul", cpuMul, coldMul, warmMul, equalGPUBatches(gpuMulOut, vectorToGPU(mulExpected))})

	cpuBitReverse := timeCPU(iters, func() {
		_ = bitReverse(left)
	})
	logCount, err := bitReverseLogCount(len(leftGPU))
	if err != nil {
		return nil, err
	}
	coldBitReverse, warmBitReverse, gpuBitReverseOut, err := timeGPUProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpBitReverseCopy, leftGPU, nil, logCount)
	})
	if err != nil {
		return nil, fmt.Errorf("bench bit_reverse: %w", err)
	}
	results = append(results, benchResult{"bit_reverse", cpuBitReverse, coldBitReverse, warmBitReverse, equalGPUBatches(gpuBitReverseOut, vectorToGPU(bitReverseExpected))})

	return results, nil
}

func makeInputs(size int) (gnarkfr.Vector, gnarkfr.Vector) {
	left := make(gnarkfr.Vector, size)
	right := make(gnarkfr.Vector, size)
	rng := mrand.New(mrand.NewSource(int64(size)*1315423911 + 1))
	for i := 0; i < size; i++ {
		left[i].SetUint64(rng.Uint64())
		right[i].SetUint64(rng.Uint64())
	}
	return left, right
}

func bitReverse(in gnarkfr.Vector) gnarkfr.Vector {
	out := make(gnarkfr.Vector, len(in))
	logCount := uint(bits.TrailingZeros(uint(len(in))))
	for i := range in {
		j := bits.Reverse(uint(i)) >> (bits.UintSize - logCount)
		out[i] = in[j]
	}
	return out
}

func vectorToGPU(in gnarkfr.Vector) []curvegpu.U32x8 {
	out := make([]curvegpu.U32x8, len(in))
	for i := range in {
		out[i] = curvegpu.SplitWords4([4]uint64(in[i]))
	}
	return out
}

func equalGPUBatches(a, b []curvegpu.U32x8) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func timeCPU(iters int, fn func()) time.Duration {
	fn()
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn()
	}
	return time.Since(start) / time.Duration(iters)
}

func timeGPUProfiled(iters int, fn func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error)) (bls12381.FrVectorProfile, bls12381.FrVectorProfile, []curvegpu.U32x8, error) {
	last, cold, err := fn()
	if err != nil {
		return bls12381.FrVectorProfile{}, bls12381.FrVectorProfile{}, nil, err
	}
	if iters == 1 {
		return cold, cold, last, nil
	}
	var warm bls12381.FrVectorProfile
	for i := 0; i < iters; i++ {
		var profile bls12381.FrVectorProfile
		last, profile, err = fn()
		if err != nil {
			return bls12381.FrVectorProfile{}, bls12381.FrVectorProfile{}, nil, err
		}
		warm.Upload += profile.Upload
		warm.Kernel += profile.Kernel
		warm.Readback += profile.Readback
		warm.Total += profile.Total
	}
	divisor := time.Duration(iters)
	warm.Upload /= divisor
	warm.Kernel /= divisor
	warm.Readback /= divisor
	warm.Total /= divisor
	return cold, warm, last, nil
}

func durationMS(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

func isResourceError(err error) bool {
	if err == nil {
		return false
	}
	text := strings.ToLower(err.Error())
	return strings.Contains(text, "buffer") ||
		strings.Contains(text, "resource") ||
		strings.Contains(text, "memory")
}

func bitReverseLogCount(count int) (uint32, error) {
	if count <= 0 {
		return 0, fmt.Errorf("bit reverse requires non-zero input")
	}
	if count&(count-1) != 0 {
		return 0, fmt.Errorf("bit reverse requires power-of-two length, got %d", count)
	}
	return uint32(bits.Len(uint(count)) - 1), nil
}
