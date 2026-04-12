package main

import (
	"flag"
	"fmt"
	"math/bits"
	mrand "math/rand"
	"time"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	"github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
	"github.com/ivokub/wnark-crypto/internal/benchutil"
)

type benchResult struct {
	name   string
	cpu    time.Duration
	cold   bn254.FrVectorProfile
	warm   bn254.FrVectorProfile
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

	fmt.Println("=== BN254 fr Vector Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", *minLog, *maxLog)
	fmt.Printf("iterations: %d\n\n", *iters)

	fmt.Println("size,op,init_ms,cpu_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_total_ms,cold_with_init_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_total_ms,verified")
	for logSize := *minLog; logSize <= *maxLog; logSize++ {
		size := 1 << logSize
		left, right := makeInputs(size)
		leftGPU := vectorToGPU(left)
		rightGPU := vectorToGPU(right)

		initStart := time.Now()
		dev, err := bn254.NewHeadlessDevice()
		if err != nil {
			panic(err)
		}
		vectorKernel, err := bn254.NewFrVectorKernel(dev.Device)
		if err != nil {
			dev.Close()
			panic(err)
		}
		initElapsed := time.Since(initStart)

		results, err := runBenchmarks(vectorKernel, left, right, leftGPU, rightGPU, *iters)
		vectorKernel.Close()
		dev.Close()
		if err != nil {
			if benchutil.IsResourceError(err) {
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
				benchutil.DurationMS(initElapsed),
				benchutil.DurationMS(result.cpu),
				benchutil.DurationMS(result.cold.Upload),
				benchutil.DurationMS(result.cold.Kernel),
				benchutil.DurationMS(result.cold.Readback),
				benchutil.DurationMS(result.cold.Total),
				benchutil.DurationMS(initElapsed+result.cold.Total),
				benchutil.DurationMS(result.warm.Upload),
				benchutil.DurationMS(result.warm.Kernel),
				benchutil.DurationMS(result.warm.Readback),
				benchutil.DurationMS(result.warm.Total),
				result.verify,
			)
		}
	}
}

func runBenchmarks(
	vectorKernel *bn254.FrVectorKernel,
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

	cpuAdd := benchutil.TimeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Add(left, right)
	})
	coldAdd, warmAdd, gpuAddOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bn254.FrVectorOpAdd, leftGPU, rightGPU, 0)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
	if err != nil {
		return nil, fmt.Errorf("bench add: %w", err)
	}
	results = append(results, benchResult{
		name:   "add",
		cpu:    cpuAdd,
		cold:   coldAdd,
		warm:   warmAdd,
		verify: equalGPUBatches(gpuAddOut, vectorToGPU(addExpected)),
	})

	cpuSub := benchutil.TimeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Sub(left, right)
	})
	coldSub, warmSub, gpuSubOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bn254.FrVectorOpSub, leftGPU, rightGPU, 0)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
	if err != nil {
		return nil, fmt.Errorf("bench sub: %w", err)
	}
	results = append(results, benchResult{
		name:   "sub",
		cpu:    cpuSub,
		cold:   coldSub,
		warm:   warmSub,
		verify: equalGPUBatches(gpuSubOut, vectorToGPU(subExpected)),
	})

	cpuMul := benchutil.TimeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Mul(left, right)
	})
	coldMul, warmMul, gpuMulOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bn254.FrVectorOpMulFactors, leftGPU, rightGPU, 0)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
	if err != nil {
		return nil, fmt.Errorf("bench mul: %w", err)
	}
	results = append(results, benchResult{
		name:   "mul",
		cpu:    cpuMul,
		cold:   coldMul,
		warm:   warmMul,
		verify: equalGPUBatches(gpuMulOut, vectorToGPU(mulExpected)),
	})

	cpuBitReverse := benchutil.TimeCPU(iters, func() {
		_ = bitReverse(left)
	})
	logCount, err := benchutil.BitReverseLogCount(len(leftGPU))
	if err != nil {
		return nil, err
	}
	coldBitReverse, warmBitReverse, gpuBitReverseOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bn254.FrVectorOpBitReverseCopy, leftGPU, nil, logCount)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
	if err != nil {
		return nil, fmt.Errorf("bench bit_reverse: %w", err)
	}
	results = append(results, benchResult{
		name:   "bit_reverse",
		cpu:    cpuBitReverse,
		cold:   coldBitReverse,
		warm:   warmBitReverse,
		verify: equalGPUBatches(gpuBitReverseOut, vectorToGPU(bitReverseExpected)),
	})

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

func zeroFrVectorProfile() bn254.FrVectorProfile {
	return bn254.FrVectorProfile{}
}

func addFrVectorProfile(dst *bn254.FrVectorProfile, src bn254.FrVectorProfile) {
	dst.Upload += src.Upload
	dst.Kernel += src.Kernel
	dst.Readback += src.Readback
	dst.Total += src.Total
}

func divFrVectorProfile(dst *bn254.FrVectorProfile, divisor int) {
	d := time.Duration(divisor)
	dst.Upload /= d
	dst.Kernel /= d
	dst.Readback /= d
	dst.Total /= d
}
