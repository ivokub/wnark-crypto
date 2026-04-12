package main

import (
	"flag"
	"fmt"
	"math/bits"
	mrand "math/rand"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	gnarkblsfr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkblsfft "github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkbnfp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	gnarkbnfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
	gnarkbnfft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/utils"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bls12381 "github.com/ivokub/wnark-crypto/go/curvegpu/bls12_381"
	bn254 "github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
	"github.com/ivokub/wnark-crypto/internal/benchutil"
)

func main() {
	var (
		curve  = flag.String("curve", "bn254", "curve: bn254 | bls12_381")
		suite  = flag.String("suite", "fr_vector", "suite: fr_vector | fr_ntt | g1_msm")
		minLog = flag.Int("min-log", 10, "minimum input size log2")
		maxLog = flag.Int("max-log", 18, "maximum input size log2")
		iters  = flag.Int("iters", 2, "iterations per benchmark")
	)
	flag.Parse()

	if *minLog < 1 || *maxLog < *minLog {
		panic("invalid min-log/max-log")
	}
	if *iters <= 0 {
		panic("iters must be positive")
	}

	switch *curve {
	case "bn254":
		switch *suite {
		case "fr_vector":
			runBN254FrVectorBench(*minLog, *maxLog, *iters)
		case "fr_ntt":
			runBN254FrNTTBench(*minLog, *maxLog, *iters)
		case "g1_msm":
			runBN254G1MSMBench(*minLog, *maxLog, *iters)
		default:
			dieUnsupported(*curve, *suite)
		}
	case "bls12_381":
		switch *suite {
		case "fr_vector":
			runBLSFrVectorBench(*minLog, *maxLog, *iters)
		case "fr_ntt":
			runBLSFrNTTBench(*minLog, *maxLog, *iters)
		default:
			dieUnsupported(*curve, *suite)
		}
	default:
		dieUnsupported(*curve, *suite)
	}
}

func dieUnsupported(curve, suite string) {
	fmt.Fprintf(os.Stderr, "unsupported benchmark combination: curve=%s suite=%s\n", curve, suite)
	fmt.Fprintln(os.Stderr, "supported:")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=fr_vector")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=fr_ntt")
	fmt.Fprintln(os.Stderr, "  -curve=bn254 -suite=g1_msm")
	fmt.Fprintln(os.Stderr, "  -curve=bls12_381 -suite=fr_vector")
	fmt.Fprintln(os.Stderr, "  -curve=bls12_381 -suite=fr_ntt")
	os.Exit(2)
}

func runBN254FrVectorBench(minLog, maxLog, iters int) {
	fmt.Println("=== BN254 fr Vector Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", minLog, maxLog)
	fmt.Printf("iterations: %d\n\n", iters)
	fmt.Println("size,op,init_ms,cpu_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_total_ms,cold_with_init_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_total_ms,verified")

	for logSize := minLog; logSize <= maxLog; logSize++ {
		size := 1 << logSize
		left, right := makeBN254VectorInputs(size)
		leftGPU := bn254VectorToGPU(left)
		rightGPU := bn254VectorToGPU(right)

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

		results, err := runBN254FrVectorBenchmarks(vectorKernel, left, right, leftGPU, rightGPU, iters)
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

func runBLSFrVectorBench(minLog, maxLog, iters int) {
	fmt.Println("=== BLS12-381 fr Vector Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", minLog, maxLog)
	fmt.Printf("iterations: %d\n\n", iters)
	fmt.Println("size,op,init_ms,cpu_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_total_ms,cold_with_init_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_total_ms,verified")

	for logSize := minLog; logSize <= maxLog; logSize++ {
		size := 1 << logSize
		left, right := makeBLSVectorInputs(size)
		leftGPU := blsVectorToGPU(left)
		rightGPU := blsVectorToGPU(right)

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

		results, err := runBLSFrVectorBenchmarks(vectorKernel, left, right, leftGPU, rightGPU, iters)
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

func runBN254FrNTTBench(minLog, maxLog, iters int) {
	fmt.Println("=== BN254 fr NTT Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", minLog, maxLog)
	fmt.Printf("iterations: %d\n\n", iters)
	fmt.Println("size,op,init_ms,cpu_ms,cold_bit_reverse_ms,cold_stage_upload_ms,cold_stage_kernel_ms,cold_stage_readback_ms,cold_stage_total_ms,cold_scale_ms,cold_total_ms,cold_with_init_ms,warm_bit_reverse_ms,warm_stage_total_ms,warm_scale_ms,warm_total_ms,verified")

	for logSize := minLog; logSize <= maxLog; logSize++ {
		size := 1 << logSize
		results, err := runBN254FrNTTSize(size, iters)
		if err != nil {
			if benchutil.IsResourceError(err) {
				fmt.Printf("# stop at size=2^%d: %v\n", logSize, err)
				break
			}
			panic(err)
		}
		for _, result := range results {
			fmt.Printf(
				"%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%t\n",
				size,
				result.Name,
				benchutil.DurationMS(result.Init),
				benchutil.DurationMS(result.CPU),
				benchutil.DurationMS(result.Cold.BitReverse.Total),
				benchutil.DurationMS(result.Cold.Stages.Upload),
				benchutil.DurationMS(result.Cold.Stages.Kernel),
				benchutil.DurationMS(result.Cold.Stages.Readback),
				benchutil.DurationMS(result.Cold.Stages.Total),
				benchutil.DurationMS(result.Cold.Scale.Total),
				benchutil.DurationMS(result.Cold.Total),
				benchutil.DurationMS(result.Init+result.Cold.Total),
				benchutil.DurationMS(result.Warm.BitReverse.Total),
				benchutil.DurationMS(result.Warm.Stages.Total),
				benchutil.DurationMS(result.Warm.Scale.Total),
				benchutil.DurationMS(result.Warm.Total),
				result.Verify,
			)
		}
	}
}

func runBLSFrNTTBench(minLog, maxLog, iters int) {
	fmt.Println("=== BLS12-381 fr NTT Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", minLog, maxLog)
	fmt.Printf("iterations: %d\n\n", iters)
	fmt.Println("size,op,init_ms,cpu_ms,cold_bit_reverse_ms,cold_stage_upload_ms,cold_stage_kernel_ms,cold_stage_readback_ms,cold_stage_total_ms,cold_scale_ms,cold_total_ms,cold_with_init_ms,warm_bit_reverse_ms,warm_stage_total_ms,warm_scale_ms,warm_total_ms,verified")

	for logSize := minLog; logSize <= maxLog; logSize++ {
		size := 1 << logSize
		results, err := runBLSFrNTTSize(size, iters)
		if err != nil {
			if benchutil.IsResourceError(err) {
				fmt.Printf("# stop at size=2^%d: %v\n", logSize, err)
				break
			}
			panic(err)
		}
		for _, result := range results {
			fmt.Printf(
				"%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%t\n",
				size,
				result.Name,
				benchutil.DurationMS(result.Init),
				benchutil.DurationMS(result.CPU),
				benchutil.DurationMS(result.Cold.BitReverse.Total),
				benchutil.DurationMS(result.Cold.Stages.Upload),
				benchutil.DurationMS(result.Cold.Stages.Kernel),
				benchutil.DurationMS(result.Cold.Stages.Readback),
				benchutil.DurationMS(result.Cold.Stages.Total),
				benchutil.DurationMS(result.Cold.Scale.Total),
				benchutil.DurationMS(result.Cold.Total),
				benchutil.DurationMS(result.Init+result.Cold.Total),
				benchutil.DurationMS(result.Warm.BitReverse.Total),
				benchutil.DurationMS(result.Warm.Stages.Total),
				benchutil.DurationMS(result.Warm.Scale.Total),
				benchutil.DurationMS(result.Warm.Total),
				result.Verify,
			)
		}
	}
}

func runBN254G1MSMBench(minLog, maxLog, iters int) {
	fmt.Println("=== BN254 G1 MSM Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", minLog, maxLog)
	fmt.Printf("iterations: %d\n\n", iters)
	fmt.Println("size,op,window,init_ms,cpu_ms,cold_partition_ms,cold_scalar_mul_ms,cold_bucket_ms,cold_window_ms,cold_final_ms,cold_reduction_ms,cold_total_ms,cold_with_init_ms,warm_partition_ms,warm_scalar_mul_ms,warm_bucket_ms,warm_window_ms,warm_final_ms,warm_reduction_ms,warm_total_ms,verified")

	for logSize := minLog; logSize <= maxLog; logSize++ {
		size := 1 << logSize
		basesCPU, scalarsCPU := makeBN254MSMInputs(size)
		basesGPU := bn254PointsToGPU(basesCPU)
		scalarsGPU := bn254ScalarsToGPU(scalarsCPU)

		benchmarks := []struct {
			label string
			run   func(*bn254.G1MSMKernel) (bn254MSMBenchResult, error)
		}{
			{
				label: "msm_naive_affine",
				run: func(kernel *bn254.G1MSMKernel) (bn254MSMBenchResult, error) {
					return runBN254MSMBenchmark(basesCPU, scalarsCPU, iters, func() ([]bn254.G1Jac, bn254.G1MSMProfile, error) {
						return kernel.RunAffineNaiveProfiled(basesGPU, scalarsGPU, len(basesGPU))
					}, 0)
				},
			},
			{
				label: "msm_pippenger_affine",
				run: func(kernel *bn254.G1MSMKernel) (bn254MSMBenchResult, error) {
					window := bn254.BestPippengerWindow(len(basesGPU))
					return runBN254MSMBenchmark(basesCPU, scalarsCPU, iters, func() ([]bn254.G1Jac, bn254.G1MSMProfile, error) {
						return kernel.RunAffinePippengerProfiled(basesGPU, scalarsGPU, len(basesGPU), window)
					}, window)
				},
			},
		}

		for _, bench := range benchmarks {
			initStart := time.Now()
			deviceSet, err := bn254.NewHeadlessDevice()
			if err != nil {
				panic(err)
			}
			kernel, err := bn254.NewG1MSMKernel(deviceSet.Device)
			if err != nil {
				deviceSet.Close()
				panic(err)
			}
			initElapsed := time.Since(initStart)

			result, err := bench.run(kernel)
			kernel.Close()
			deviceSet.Close()
			if err != nil {
				if benchutil.IsResourceError(err) {
					fmt.Printf("# stop at size=2^%d op=%s: %v\n", logSize, bench.label, err)
					return
				}
				panic(err)
			}

			fmt.Printf(
				"%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%t\n",
				size,
				bench.label,
				result.window,
				benchutil.DurationMS(initElapsed),
				benchutil.DurationMS(result.cpu),
				benchutil.DurationMS(result.cold.Partition),
				benchutil.DurationMS(result.cold.ScalarMul),
				benchutil.DurationMS(result.cold.BucketReduction),
				benchutil.DurationMS(result.cold.WindowReduction),
				benchutil.DurationMS(result.cold.FinalReduction),
				benchutil.DurationMS(result.cold.Reduction),
				benchutil.DurationMS(result.cold.Total),
				benchutil.DurationMS(initElapsed+result.cold.Total),
				benchutil.DurationMS(result.warm.Partition),
				benchutil.DurationMS(result.warm.ScalarMul),
				benchutil.DurationMS(result.warm.BucketReduction),
				benchutil.DurationMS(result.warm.WindowReduction),
				benchutil.DurationMS(result.warm.FinalReduction),
				benchutil.DurationMS(result.warm.Reduction),
				benchutil.DurationMS(result.warm.Total),
				result.verify,
			)
		}
	}
}

func runBN254FrVectorBenchmarks(
	vectorKernel *bn254.FrVectorKernel,
	left, right gnarkbnfr.Vector,
	leftGPU, rightGPU []curvegpu.U32x8,
	iters int,
) ([]benchutil.VectorBenchResult[bn254.FrVectorProfile], error) {
	addExpected := make(gnarkbnfr.Vector, len(left))
	addExpected.Add(left, right)
	subExpected := make(gnarkbnfr.Vector, len(left))
	subExpected.Sub(left, right)
	mulExpected := make(gnarkbnfr.Vector, len(left))
	mulExpected.Mul(left, right)
	bitReverseExpected := bn254BitReverse(left)

	logCount, err := benchutil.BitReverseLogCount(len(leftGPU))
	if err != nil {
		return nil, err
	}
	return benchutil.RunVectorBenchmarks(iters, []benchutil.VectorBenchCase[bn254.FrVectorProfile]{
		{
			Name: "add",
			CPU:  func() { out := make(gnarkbnfr.Vector, len(left)); out.Add(left, right) },
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpAdd, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench add: %w", err)
				}
				return out, profile, nil
			},
			Expected: bn254VectorToGPU(addExpected),
		},
		{
			Name: "sub",
			CPU:  func() { out := make(gnarkbnfr.Vector, len(left)); out.Sub(left, right) },
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpSub, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench sub: %w", err)
				}
				return out, profile, nil
			},
			Expected: bn254VectorToGPU(subExpected),
		},
		{
			Name: "mul",
			CPU:  func() { out := make(gnarkbnfr.Vector, len(left)); out.Mul(left, right) },
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpMulFactors, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench mul: %w", err)
				}
				return out, profile, nil
			},
			Expected: bn254VectorToGPU(mulExpected),
		},
		{
			Name: "bit_reverse",
			CPU:  func() { _ = bn254BitReverse(left) },
			GPU: func() ([]curvegpu.U32x8, bn254.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bn254.FrVectorOpBitReverseCopy, leftGPU, nil, logCount)
				if err != nil {
					return nil, bn254.FrVectorProfile{}, fmt.Errorf("bench bit_reverse: %w", err)
				}
				return out, profile, nil
			},
			Expected: bn254VectorToGPU(bitReverseExpected),
		},
	}, zeroBN254FrVectorProfile, addBN254FrVectorProfile, divBN254FrVectorProfile)
}

func runBLSFrVectorBenchmarks(
	vectorKernel *bls12381.FrVectorKernel,
	left, right gnarkblsfr.Vector,
	leftGPU, rightGPU []curvegpu.U32x8,
	iters int,
) ([]benchutil.VectorBenchResult[bls12381.FrVectorProfile], error) {
	addExpected := make(gnarkblsfr.Vector, len(left))
	addExpected.Add(left, right)
	subExpected := make(gnarkblsfr.Vector, len(left))
	subExpected.Sub(left, right)
	mulExpected := make(gnarkblsfr.Vector, len(left))
	mulExpected.Mul(left, right)
	bitReverseExpected := blsBitReverse(left)

	logCount, err := benchutil.BitReverseLogCount(len(leftGPU))
	if err != nil {
		return nil, err
	}
	return benchutil.RunVectorBenchmarks(iters, []benchutil.VectorBenchCase[bls12381.FrVectorProfile]{
		{
			Name: "add",
			CPU:  func() { out := make(gnarkblsfr.Vector, len(left)); out.Add(left, right) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bls12381.FrVectorOpAdd, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bls12381.FrVectorProfile{}, fmt.Errorf("bench add: %w", err)
				}
				return out, profile, nil
			},
			Expected: blsVectorToGPU(addExpected),
		},
		{
			Name: "sub",
			CPU:  func() { out := make(gnarkblsfr.Vector, len(left)); out.Sub(left, right) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bls12381.FrVectorOpSub, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bls12381.FrVectorProfile{}, fmt.Errorf("bench sub: %w", err)
				}
				return out, profile, nil
			},
			Expected: blsVectorToGPU(subExpected),
		},
		{
			Name: "mul",
			CPU:  func() { out := make(gnarkblsfr.Vector, len(left)); out.Mul(left, right) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bls12381.FrVectorOpMulFactors, leftGPU, rightGPU, 0)
				if err != nil {
					return nil, bls12381.FrVectorProfile{}, fmt.Errorf("bench mul: %w", err)
				}
				return out, profile, nil
			},
			Expected: blsVectorToGPU(mulExpected),
		},
		{
			Name: "bit_reverse",
			CPU:  func() { _ = blsBitReverse(left) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrVectorProfile, error) {
				out, profile, err := vectorKernel.RunProfiled(bls12381.FrVectorOpBitReverseCopy, leftGPU, nil, logCount)
				if err != nil {
					return nil, bls12381.FrVectorProfile{}, fmt.Errorf("bench bit_reverse: %w", err)
				}
				return out, profile, nil
			},
			Expected: blsVectorToGPU(bitReverseExpected),
		},
	}, zeroBLSFrVectorProfile, addBLSFrVectorProfile, divBLSFrVectorProfile)
}

func runBN254FrNTTSize(size, iters int) ([]benchutil.NTTBenchResult[bn254.FrNTTProfile], error) {
	domain := gnarkbnfft.NewDomain(uint64(size))
	stageTwiddles, inverseStageTwiddles, scale, err := buildBN254Twiddles(domain, size)
	if err != nil {
		return nil, err
	}

	input := makeBN254NTTInput(size)
	forwardExpected := bn254CPUForward(domain, input)
	inverseExpected := bn254CPUInverse(domain, forwardExpected)

	inputGPU := bn254VectorToGPU(input)
	forwardExpectedGPU := bn254VectorToGPU(forwardExpected)

	forwardKernel, forwardInit, err := newBN254NTTKernel()
	if err != nil {
		return nil, err
	}
	defer forwardKernel.close()

	inverseKernel, inverseInit, err := newBN254NTTKernel()
	if err != nil {
		return nil, err
	}
	defer inverseKernel.close()

	return benchutil.RunNTTBenchmarks(iters, []benchutil.NTTBenchCase[bn254.FrNTTProfile]{
		{
			Name: "forward_ntt",
			Init: forwardInit,
			CPU:  func() { _ = bn254CPUForward(domain, input) },
			GPU: func() ([]curvegpu.U32x8, bn254.FrNTTProfile, error) {
				out, profile, err := forwardKernel.kernel.ForwardDITFullPathProfiled(inputGPU, stageTwiddles)
				if err != nil {
					return nil, bn254.FrNTTProfile{}, fmt.Errorf("bench forward_ntt: %w", err)
				}
				return out, profile, nil
			},
			Expected: bn254VectorToGPU(forwardExpected),
		},
		{
			Name: "inverse_ntt",
			Init: inverseInit,
			CPU:  func() { _ = bn254CPUInverse(domain, forwardExpected) },
			GPU: func() ([]curvegpu.U32x8, bn254.FrNTTProfile, error) {
				out, profile, err := inverseKernel.kernel.InverseDITFullPathProfiled(forwardExpectedGPU, inverseStageTwiddles, scale)
				if err != nil {
					return nil, bn254.FrNTTProfile{}, fmt.Errorf("bench inverse_ntt: %w", err)
				}
				return out, profile, nil
			},
			Expected: bn254VectorToGPU(inverseExpected),
		},
	}, zeroBN254FrNTTProfile, addBN254FrNTTProfile, divBN254FrNTTProfile)
}

func runBLSFrNTTSize(size, iters int) ([]benchutil.NTTBenchResult[bls12381.FrNTTProfile], error) {
	domain := gnarkblsfft.NewDomain(uint64(size))
	stageTwiddles, inverseStageTwiddles, scale, err := buildBLSTwiddles(domain)
	if err != nil {
		return nil, err
	}

	input := makeBLSNTTInput(size)
	forwardExpected := blsCPUForward(domain, input)
	inverseExpected := blsCPUInverse(domain, forwardExpected)

	inputGPU := blsVectorToGPU(input)
	forwardExpectedGPU := blsVectorToGPU(forwardExpected)

	forwardKernel, forwardInit, err := newBLSNTTKernel()
	if err != nil {
		return nil, err
	}
	defer forwardKernel.close()

	inverseKernel, inverseInit, err := newBLSNTTKernel()
	if err != nil {
		return nil, err
	}
	defer inverseKernel.close()

	return benchutil.RunNTTBenchmarks(iters, []benchutil.NTTBenchCase[bls12381.FrNTTProfile]{
		{
			Name: "forward_ntt",
			Init: forwardInit,
			CPU:  func() { _ = blsCPUForward(domain, input) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrNTTProfile, error) {
				out, profile, err := forwardKernel.kernel.ForwardDITFullPathProfiled(inputGPU, stageTwiddles)
				if err != nil {
					return nil, bls12381.FrNTTProfile{}, fmt.Errorf("bench forward_ntt: %w", err)
				}
				return out, profile, nil
			},
			Expected: blsVectorToGPU(forwardExpected),
		},
		{
			Name: "inverse_ntt",
			Init: inverseInit,
			CPU:  func() { _ = blsCPUInverse(domain, forwardExpected) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrNTTProfile, error) {
				out, profile, err := inverseKernel.kernel.InverseDITFullPathProfiled(forwardExpectedGPU, inverseStageTwiddles, scale)
				if err != nil {
					return nil, bls12381.FrNTTProfile{}, fmt.Errorf("bench inverse_ntt: %w", err)
				}
				return out, profile, nil
			},
			Expected: blsVectorToGPU(inverseExpected),
		},
	}, zeroBLSFrNTTProfile, addBLSFrNTTProfile, divBLSFrNTTProfile)
}

type bn254MSMBenchResult struct {
	cpu    time.Duration
	cold   bn254.G1MSMProfile
	warm   bn254.G1MSMProfile
	verify bool
	window uint32
}

func runBN254MSMBenchmark(
	basesCPU []gnarkbn254.G1Affine,
	scalarsCPU []gnarkbnfr.Element,
	iters int,
	run func() ([]bn254.G1Jac, bn254.G1MSMProfile, error),
	window uint32,
) (bn254MSMBenchResult, error) {
	cpuResult := benchutil.TimeCPU(iters, func() {
		var out gnarkbn254.G1Affine
		if _, err := out.MultiExp(basesCPU, scalarsCPU, ecc.MultiExpConfig{}); err != nil {
			panic(err)
		}
	})

	cold, warm, gpuOut, err := averageBN254MSMProfile(iters, run)
	if err != nil {
		return bn254MSMBenchResult{}, err
	}

	var want gnarkbn254.G1Affine
	if _, err := want.MultiExp(basesCPU, scalarsCPU, ecc.MultiExpConfig{}); err != nil {
		return bn254MSMBenchResult{}, fmt.Errorf("cpu reference: %w", err)
	}

	return bn254MSMBenchResult{
		cpu:    cpuResult,
		cold:   cold,
		warm:   warm,
		verify: len(gpuOut) == 1 && equalBN254GPUToCPU(gpuOut[0], want),
		window: window,
	}, nil
}

func averageBN254MSMProfile(
	iters int,
	fn func() ([]bn254.G1Jac, bn254.G1MSMProfile, error),
) (bn254.G1MSMProfile, bn254.G1MSMProfile, []bn254.G1Jac, error) {
	last, cold, err := fn()
	if err != nil {
		return bn254.G1MSMProfile{}, bn254.G1MSMProfile{}, nil, err
	}
	if iters == 1 {
		return cold, cold, last, nil
	}
	var warm bn254.G1MSMProfile
	for i := 0; i < iters; i++ {
		var profile bn254.G1MSMProfile
		last, profile, err = fn()
		if err != nil {
			return bn254.G1MSMProfile{}, bn254.G1MSMProfile{}, nil, err
		}
		warm.Partition += profile.Partition
		warm.ScalarMul += profile.ScalarMul
		warm.BucketReduction += profile.BucketReduction
		warm.WindowReduction += profile.WindowReduction
		warm.FinalReduction += profile.FinalReduction
		warm.Reduction += profile.Reduction
		warm.Total += profile.Total
	}
	d := time.Duration(iters)
	warm.Partition /= d
	warm.ScalarMul /= d
	warm.BucketReduction /= d
	warm.WindowReduction /= d
	warm.FinalReduction /= d
	warm.Reduction /= d
	warm.Total /= d
	return cold, warm, last, nil
}

type bn254NTTKernelSet struct {
	device *bn254.DeviceSet
	kernel *bn254.FrNTTKernel
}

func (k bn254NTTKernelSet) close() {
	if k.kernel != nil {
		k.kernel.Close()
	}
	if k.device != nil {
		k.device.Close()
	}
}

func newBN254NTTKernel() (bn254NTTKernelSet, time.Duration, error) {
	start := time.Now()
	deviceSet, err := bn254.NewHeadlessDevice()
	if err != nil {
		return bn254NTTKernelSet{}, 0, err
	}
	kernel, err := bn254.NewFrNTTKernel(deviceSet.Device)
	if err != nil {
		deviceSet.Close()
		return bn254NTTKernelSet{}, 0, err
	}
	return bn254NTTKernelSet{device: deviceSet, kernel: kernel}, time.Since(start), nil
}

type blsNTTKernelSet struct {
	device *bls12381.DeviceSet
	kernel *bls12381.FrNTTKernel
}

func (k blsNTTKernelSet) close() {
	if k.kernel != nil {
		k.kernel.Close()
	}
	if k.device != nil {
		k.device.Close()
	}
}

func newBLSNTTKernel() (blsNTTKernelSet, time.Duration, error) {
	start := time.Now()
	deviceSet, err := bls12381.NewHeadlessDevice()
	if err != nil {
		return blsNTTKernelSet{}, 0, err
	}
	kernel, err := bls12381.NewFrNTTKernel(deviceSet.Device)
	if err != nil {
		deviceSet.Close()
		return blsNTTKernelSet{}, 0, err
	}
	return blsNTTKernelSet{device: deviceSet, kernel: kernel}, time.Since(start), nil
}

func makeBN254VectorInputs(size int) (gnarkbnfr.Vector, gnarkbnfr.Vector) {
	left := make(gnarkbnfr.Vector, size)
	right := make(gnarkbnfr.Vector, size)
	rng := mrand.New(mrand.NewSource(int64(size)*1315423911 + 1))
	for i := 0; i < size; i++ {
		left[i].SetUint64(rng.Uint64())
		right[i].SetUint64(rng.Uint64())
	}
	return left, right
}

func makeBLSVectorInputs(size int) (gnarkblsfr.Vector, gnarkblsfr.Vector) {
	left := make(gnarkblsfr.Vector, size)
	right := make(gnarkblsfr.Vector, size)
	rng := mrand.New(mrand.NewSource(int64(size)*1315423911 + 1))
	for i := 0; i < size; i++ {
		left[i].SetUint64(rng.Uint64())
		right[i].SetUint64(rng.Uint64())
	}
	return left, right
}

func makeBN254NTTInput(size int) gnarkbnfr.Vector {
	out := make(gnarkbnfr.Vector, size)
	rng := mrand.New(mrand.NewSource(int64(size)*2654435761 + 7))
	for i := range out {
		out[i].SetUint64(rng.Uint64())
	}
	return out
}

func makeBLSNTTInput(size int) gnarkblsfr.Vector {
	out := make(gnarkblsfr.Vector, size)
	rng := mrand.New(mrand.NewSource(int64(size)*2654435761 + 7))
	for i := range out {
		out[i].SetUint64(rng.Uint64())
	}
	return out
}

func makeBN254MSMInputs(size int) ([]gnarkbn254.G1Affine, []gnarkbnfr.Element) {
	bases := make([]gnarkbn254.G1Affine, size)
	scalars := make([]gnarkbnfr.Element, size)
	rng := mrand.New(mrand.NewSource(int64(size)*2654435761 + 17))

	g1Jac, _, _, _ := gnarkbn254.Generators()
	var acc gnarkbn254.G1Jac
	acc.Set(&g1Jac)
	for i := 0; i < size; i++ {
		bases[i].FromJacobian(&acc)
		acc.AddAssign(&g1Jac)
		scalars[i].SetUint64(rng.Uint64())
	}
	return bases, scalars
}

func bn254BitReverse(in gnarkbnfr.Vector) gnarkbnfr.Vector {
	out := make(gnarkbnfr.Vector, len(in))
	logCount := uint(bits.TrailingZeros(uint(len(in))))
	for i := range in {
		j := bits.Reverse(uint(i)) >> (bits.UintSize - logCount)
		out[i] = in[j]
	}
	return out
}

func blsBitReverse(in gnarkblsfr.Vector) gnarkblsfr.Vector {
	out := make(gnarkblsfr.Vector, len(in))
	logCount := uint(bits.TrailingZeros(uint(len(in))))
	for i := range in {
		j := bits.Reverse(uint(i)) >> (bits.UintSize - logCount)
		out[i] = in[j]
	}
	return out
}

func bn254CPUForward(domain *gnarkbnfft.Domain, input gnarkbnfr.Vector) gnarkbnfr.Vector {
	state := make(gnarkbnfr.Vector, len(input))
	copy(state, input)
	utils.BitReverse(state)
	domain.FFT(state, gnarkbnfft.DIT)
	return state
}

func bn254CPUInverse(domain *gnarkbnfft.Domain, input gnarkbnfr.Vector) gnarkbnfr.Vector {
	state := make(gnarkbnfr.Vector, len(input))
	copy(state, input)
	utils.BitReverse(state)
	domain.FFTInverse(state, gnarkbnfft.DIT)
	return state
}

func blsCPUForward(domain *gnarkblsfft.Domain, input gnarkblsfr.Vector) gnarkblsfr.Vector {
	state := make(gnarkblsfr.Vector, len(input))
	copy(state, input)
	utils.BitReverse(state)
	domain.FFT(state, gnarkblsfft.DIT)
	return state
}

func blsCPUInverse(domain *gnarkblsfft.Domain, input gnarkblsfr.Vector) gnarkblsfr.Vector {
	state := make(gnarkblsfr.Vector, len(input))
	copy(state, input)
	utils.BitReverse(state)
	domain.FFTInverse(state, gnarkblsfft.DIT)
	return state
}

func buildBN254Twiddles(domain *gnarkbnfft.Domain, size int) ([][]curvegpu.U32x8, [][]curvegpu.U32x8, curvegpu.U32x8, error) {
	twiddles, err := domain.Twiddles()
	if err != nil {
		return nil, nil, curvegpu.U32x8{}, err
	}
	twiddlesInv, err := domain.TwiddlesInv()
	if err != nil {
		return nil, nil, curvegpu.U32x8{}, err
	}
	logN := len(twiddles)
	stageTwiddles := make([][]curvegpu.U32x8, logN)
	inverseStageTwiddles := make([][]curvegpu.U32x8, logN)
	for stage := 1; stage <= logN; stage++ {
		m := 1 << (stage - 1)
		stageTwiddles[stage-1] = bn254VectorToGPU(gnarkbnfr.Vector(twiddles[logN-stage][:m]))
		inverseStageTwiddles[stage-1] = bn254VectorToGPU(gnarkbnfr.Vector(twiddlesInv[logN-stage][:m]))
	}
	return stageTwiddles, inverseStageTwiddles, curvegpu.SplitWords4([4]uint64(domain.CardinalityInv)), nil
}

func buildBLSTwiddles(domain *gnarkblsfft.Domain) ([][]curvegpu.U32x8, [][]curvegpu.U32x8, curvegpu.U32x8, error) {
	twiddles, err := domain.Twiddles()
	if err != nil {
		return nil, nil, curvegpu.U32x8{}, err
	}
	twiddlesInv, err := domain.TwiddlesInv()
	if err != nil {
		return nil, nil, curvegpu.U32x8{}, err
	}
	logN := len(twiddles)
	stageTwiddles := make([][]curvegpu.U32x8, logN)
	inverseStageTwiddles := make([][]curvegpu.U32x8, logN)
	for stage := 1; stage <= logN; stage++ {
		m := 1 << (stage - 1)
		stageTwiddles[stage-1] = blsVectorToGPU(gnarkblsfr.Vector(twiddles[logN-stage][:m]))
		inverseStageTwiddles[stage-1] = blsVectorToGPU(gnarkblsfr.Vector(twiddlesInv[logN-stage][:m]))
	}
	return stageTwiddles, inverseStageTwiddles, curvegpu.SplitWords4([4]uint64(domain.CardinalityInv)), nil
}

func bn254VectorToGPU(in gnarkbnfr.Vector) []curvegpu.U32x8 {
	out := make([]curvegpu.U32x8, len(in))
	for i := range in {
		out[i] = curvegpu.SplitWords4([4]uint64(in[i]))
	}
	return out
}

func blsVectorToGPU(in gnarkblsfr.Vector) []curvegpu.U32x8 {
	out := make([]curvegpu.U32x8, len(in))
	for i := range in {
		out[i] = curvegpu.SplitWords4([4]uint64(in[i]))
	}
	return out
}

func bn254PointsToGPU(in []gnarkbn254.G1Affine) []bn254.G1Affine {
	out := make([]bn254.G1Affine, len(in))
	for i := range in {
		out[i] = bn254.G1Affine{
			X: curvegpu.SplitWords4([4]uint64(in[i].X)),
			Y: curvegpu.SplitWords4([4]uint64(in[i].Y)),
		}
	}
	return out
}

func bn254ScalarsToGPU(in []gnarkbnfr.Element) []curvegpu.U32x8 {
	out := make([]curvegpu.U32x8, len(in))
	for i := range in {
		out[i] = bn254ScalarToRegularWords(in[i])
	}
	return out
}

func bn254ScalarToRegularWords(v gnarkbnfr.Element) curvegpu.U32x8 {
	bytesBE := v.Bytes()
	var out curvegpu.U32x8
	for i := 0; i < 8; i++ {
		base := len(bytesBE) - ((i + 1) * 4)
		out[i] = uint32(bytesBE[base+3]) |
			(uint32(bytesBE[base+2]) << 8) |
			(uint32(bytesBE[base+1]) << 16) |
			(uint32(bytesBE[base]) << 24)
	}
	return out
}

func equalBN254GPUToCPU(gpu bn254.G1Jac, want gnarkbn254.G1Affine) bool {
	gotJac := gnarkbn254.G1Jac{
		X: gnarkbnfp.Element(curvegpu.JoinWords8(gpu.X)),
		Y: gnarkbnfp.Element(curvegpu.JoinWords8(gpu.Y)),
		Z: gnarkbnfp.Element(curvegpu.JoinWords8(gpu.Z)),
	}
	var gotAff gnarkbn254.G1Affine
	gotAff.FromJacobian(&gotJac)
	return gotAff.Equal(&want)
}

func zeroBN254FrVectorProfile() bn254.FrVectorProfile { return bn254.FrVectorProfile{} }
func addBN254FrVectorProfile(dst *bn254.FrVectorProfile, src bn254.FrVectorProfile) {
	dst.Upload += src.Upload
	dst.Kernel += src.Kernel
	dst.Readback += src.Readback
	dst.Total += src.Total
}
func divBN254FrVectorProfile(dst *bn254.FrVectorProfile, divisor int) {
	d := time.Duration(divisor)
	dst.Upload /= d
	dst.Kernel /= d
	dst.Readback /= d
	dst.Total /= d
}

func zeroBLSFrVectorProfile() bls12381.FrVectorProfile { return bls12381.FrVectorProfile{} }
func addBLSFrVectorProfile(dst *bls12381.FrVectorProfile, src bls12381.FrVectorProfile) {
	dst.Upload += src.Upload
	dst.Kernel += src.Kernel
	dst.Readback += src.Readback
	dst.Total += src.Total
}
func divBLSFrVectorProfile(dst *bls12381.FrVectorProfile, divisor int) {
	d := time.Duration(divisor)
	dst.Upload /= d
	dst.Kernel /= d
	dst.Readback /= d
	dst.Total /= d
}

func zeroBN254FrNTTProfile() bn254.FrNTTProfile { return bn254.FrNTTProfile{} }
func addBN254FrNTTProfile(dst *bn254.FrNTTProfile, src bn254.FrNTTProfile) {
	dst.BitReverse.Upload += src.BitReverse.Upload
	dst.BitReverse.Kernel += src.BitReverse.Kernel
	dst.BitReverse.Readback += src.BitReverse.Readback
	dst.BitReverse.Total += src.BitReverse.Total
	dst.Stages.Upload += src.Stages.Upload
	dst.Stages.Kernel += src.Stages.Kernel
	dst.Stages.Readback += src.Stages.Readback
	dst.Stages.Total += src.Stages.Total
	dst.Scale.Upload += src.Scale.Upload
	dst.Scale.Kernel += src.Scale.Kernel
	dst.Scale.Readback += src.Scale.Readback
	dst.Scale.Total += src.Scale.Total
	dst.Total += src.Total
}
func divBN254FrNTTProfile(dst *bn254.FrNTTProfile, divisor int) {
	d := time.Duration(divisor)
	dst.BitReverse.Upload /= d
	dst.BitReverse.Kernel /= d
	dst.BitReverse.Readback /= d
	dst.BitReverse.Total /= d
	dst.Stages.Upload /= d
	dst.Stages.Kernel /= d
	dst.Stages.Readback /= d
	dst.Stages.Total /= d
	dst.Scale.Upload /= d
	dst.Scale.Kernel /= d
	dst.Scale.Readback /= d
	dst.Scale.Total /= d
	dst.Total /= d
}

func zeroBLSFrNTTProfile() bls12381.FrNTTProfile { return bls12381.FrNTTProfile{} }
func addBLSFrNTTProfile(dst *bls12381.FrNTTProfile, src bls12381.FrNTTProfile) {
	dst.BitReverse.Upload += src.BitReverse.Upload
	dst.BitReverse.Kernel += src.BitReverse.Kernel
	dst.BitReverse.Readback += src.BitReverse.Readback
	dst.BitReverse.Total += src.BitReverse.Total
	dst.Stages.Upload += src.Stages.Upload
	dst.Stages.Kernel += src.Stages.Kernel
	dst.Stages.Readback += src.Stages.Readback
	dst.Stages.Total += src.Stages.Total
	dst.Scale.Upload += src.Scale.Upload
	dst.Scale.Kernel += src.Scale.Kernel
	dst.Scale.Readback += src.Scale.Readback
	dst.Scale.Total += src.Scale.Total
	dst.Total += src.Total
}
func divBLSFrNTTProfile(dst *bls12381.FrNTTProfile, divisor int) {
	d := time.Duration(divisor)
	dst.BitReverse.Upload /= d
	dst.BitReverse.Kernel /= d
	dst.BitReverse.Readback /= d
	dst.BitReverse.Total /= d
	dst.Stages.Upload /= d
	dst.Stages.Kernel /= d
	dst.Stages.Readback /= d
	dst.Stages.Total /= d
	dst.Scale.Upload /= d
	dst.Scale.Kernel /= d
	dst.Scale.Readback /= d
	dst.Scale.Total /= d
	dst.Total /= d
}
