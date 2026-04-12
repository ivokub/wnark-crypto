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
				result.Name,
				benchutil.DurationMS(initElapsed),
				benchutil.DurationMS(result.CPU),
				benchutil.DurationMS(result.Cold.Upload),
				benchutil.DurationMS(result.Cold.Kernel),
				benchutil.DurationMS(result.Cold.Readback),
				benchutil.DurationMS(result.Cold.Total),
				benchutil.DurationMS(initElapsed+result.Cold.Total),
				benchutil.DurationMS(result.Warm.Upload),
				benchutil.DurationMS(result.Warm.Kernel),
				benchutil.DurationMS(result.Warm.Readback),
				benchutil.DurationMS(result.Warm.Total),
				result.Verify,
			)
		}
	}
}

func runBenchmarks(
	vectorKernel *bn254.FrVectorKernel,
	left, right gnarkfr.Vector,
	leftGPU, rightGPU []curvegpu.U32x8,
	iters int,
) ([]benchutil.VectorBenchResult[bn254.FrVectorProfile], error) {
	addExpected := make(gnarkfr.Vector, len(left))
	addExpected.Add(left, right)
	subExpected := make(gnarkfr.Vector, len(left))
	subExpected.Sub(left, right)
	mulExpected := make(gnarkfr.Vector, len(left))
	mulExpected.Mul(left, right)
	bitReverseExpected := bitReverse(left)

	logCount, err := benchutil.BitReverseLogCount(len(leftGPU))
	if err != nil {
		return nil, err
	}
	return benchutil.RunVectorBenchmarks(iters, []benchutil.VectorBenchCase[bn254.FrVectorProfile]{
		{
			Name: "add",
			CPU: func() {
				out := make(gnarkfr.Vector, len(left))
				out.Add(left, right)
			},
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpAdd, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench add: %w", err)
				}
				return out, profile, nil
			},
			Expected: vectorToGPU(addExpected),
		},
		{
			Name: "sub",
			CPU: func() {
				out := make(gnarkfr.Vector, len(left))
				out.Sub(left, right)
			},
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpSub, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench sub: %w", err)
				}
				return out, profile, nil
			},
			Expected: vectorToGPU(subExpected),
		},
		{
			Name: "mul",
			CPU: func() {
				out := make(gnarkfr.Vector, len(left))
				out.Mul(left, right)
			},
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpMulFactors, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench mul: %w", err)
				}
				return out, profile, nil
			},
			Expected: vectorToGPU(mulExpected),
		},
		{
			Name: "bit_reverse",
			CPU: func() {
				_ = bitReverse(left)
			},
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpBitReverseCopy, leftGPU, nil, logCount)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench bit_reverse: %w", err)
				}
				return out, profile, nil
			},
			Expected: vectorToGPU(bitReverseExpected),
		},
	}, zeroFrVectorProfile, addFrVectorProfile, divFrVectorProfile)
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
