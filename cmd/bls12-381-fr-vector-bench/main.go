package main

import (
	"flag"
	"fmt"
	"math/bits"
	mrand "math/rand"
	"time"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bls12381 "github.com/ivokub/wnark-crypto/go/curvegpu/bls12_381"
	"github.com/ivokub/wnark-crypto/internal/benchutil"
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

	cpuAdd := benchutil.TimeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Add(left, right)
	})
	coldAdd, warmAdd, gpuAddOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpAdd, leftGPU, rightGPU, 0)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
	if err != nil {
		return nil, fmt.Errorf("bench add: %w", err)
	}
	results = append(results, benchResult{"add", cpuAdd, coldAdd, warmAdd, equalGPUBatches(gpuAddOut, vectorToGPU(addExpected))})

	cpuSub := benchutil.TimeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Sub(left, right)
	})
	coldSub, warmSub, gpuSubOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpSub, leftGPU, rightGPU, 0)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
	if err != nil {
		return nil, fmt.Errorf("bench sub: %w", err)
	}
	results = append(results, benchResult{"sub", cpuSub, coldSub, warmSub, equalGPUBatches(gpuSubOut, vectorToGPU(subExpected))})

	cpuMul := benchutil.TimeCPU(iters, func() {
		out := make(gnarkfr.Vector, len(left))
		out.Mul(left, right)
	})
	coldMul, warmMul, gpuMulOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpMulFactors, leftGPU, rightGPU, 0)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
	if err != nil {
		return nil, fmt.Errorf("bench mul: %w", err)
	}
	results = append(results, benchResult{"mul", cpuMul, coldMul, warmMul, equalGPUBatches(gpuMulOut, vectorToGPU(mulExpected))})

	cpuBitReverse := benchutil.TimeCPU(iters, func() {
		_ = bitReverse(left)
	})
	logCount, err := benchutil.BitReverseLogCount(len(leftGPU))
	if err != nil {
		return nil, err
	}
	coldBitReverse, warmBitReverse, gpuBitReverseOut, err := benchutil.AverageProfiled(iters, func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
		return vectorKernel.RunProfiled(bls12381.FrVectorOpBitReverseCopy, leftGPU, nil, logCount)
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
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

func zeroFrVectorProfile() bls12381.FrVectorProfile {
	return bls12381.FrVectorProfile{}
}

func addFrVectorProfile(dst *bls12381.FrVectorProfile, src bls12381.FrVectorProfile) {
	dst.Upload += src.Upload
	dst.Kernel += src.Kernel
	dst.Readback += src.Readback
	dst.Total += src.Total
}

func divFrVectorProfile(dst *bls12381.FrVectorProfile, divisor int) {
	d := time.Duration(divisor)
	dst.Upload /= d
	dst.Kernel /= d
	dst.Readback /= d
	dst.Total /= d
}
