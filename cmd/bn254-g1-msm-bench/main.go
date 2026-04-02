package main

import (
	"flag"
	"fmt"
	mrand "math/rand"
	"strings"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkfp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	"github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type benchResult struct {
	cpu    time.Duration
	cold   bn254.G1MSMProfile
	warm   bn254.G1MSMProfile
	verify bool
	window uint32
}

func main() {
	var (
		minLog = flag.Int("min-log", 10, "minimum MSM size log2")
		maxLog = flag.Int("max-log", 16, "maximum MSM size log2")
		iters  = flag.Int("iters", 2, "iterations per benchmark")
	)
	flag.Parse()

	if *minLog < 1 || *maxLog < *minLog {
		panic("invalid min-log/max-log")
	}
	if *iters <= 0 {
		panic("iters must be positive")
	}

	fmt.Println("=== BN254 G1 MSM Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", *minLog, *maxLog)
	fmt.Printf("iterations: %d\n\n", *iters)
	fmt.Println("size,op,window,init_ms,cpu_ms,cold_partition_ms,cold_scalar_mul_ms,cold_bucket_ms,cold_window_ms,cold_final_ms,cold_reduction_ms,cold_total_ms,cold_with_init_ms,warm_partition_ms,warm_scalar_mul_ms,warm_bucket_ms,warm_window_ms,warm_final_ms,warm_reduction_ms,warm_total_ms,verified")

	for logSize := *minLog; logSize <= *maxLog; logSize++ {
		size := 1 << logSize
		basesCPU, scalarsCPU := makeInputs(size)
		basesGPU := pointsToGPU(basesCPU)
		scalarsGPU := scalarsToGPU(scalarsCPU)
		benchmarks := []struct {
			label string
			run   func(*bn254.G1MSMKernel) (benchResult, error)
		}{
			{
				label: "msm_naive_affine",
				run: func(kernel *bn254.G1MSMKernel) (benchResult, error) {
					return runBenchmark(basesCPU, scalarsCPU, *iters, func() ([]bn254.G1Jac, bn254.G1MSMProfile, error) {
						return kernel.RunAffineNaiveProfiled(basesGPU, scalarsGPU, len(basesGPU))
					}, 0)
				},
			},
			{
				label: "msm_pippenger_affine",
				run: func(kernel *bn254.G1MSMKernel) (benchResult, error) {
					window := bn254.BestPippengerWindow(len(basesGPU))
					return runBenchmark(basesCPU, scalarsCPU, *iters, func() ([]bn254.G1Jac, bn254.G1MSMProfile, error) {
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
				if isResourceError(err) {
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
				durationMS(initElapsed),
				durationMS(result.cpu),
				durationMS(result.cold.Partition),
				durationMS(result.cold.ScalarMul),
				durationMS(result.cold.BucketReduction),
				durationMS(result.cold.WindowReduction),
				durationMS(result.cold.FinalReduction),
				durationMS(result.cold.Reduction),
				durationMS(result.cold.Total),
				durationMS(initElapsed+result.cold.Total),
				durationMS(result.warm.Partition),
				durationMS(result.warm.ScalarMul),
				durationMS(result.warm.BucketReduction),
				durationMS(result.warm.WindowReduction),
				durationMS(result.warm.FinalReduction),
				durationMS(result.warm.Reduction),
				durationMS(result.warm.Total),
				result.verify,
			)
		}
	}
}

func runBenchmark(
	basesCPU []gnarkbn254.G1Affine,
	scalarsCPU []gnarkfr.Element,
	iters int,
	run func() ([]bn254.G1Jac, bn254.G1MSMProfile, error),
	window uint32,
) (benchResult, error) {
	cpuResult := timeCPU(iters, func() {
		var out gnarkbn254.G1Affine
		if _, err := out.MultiExp(basesCPU, scalarsCPU, ecc.MultiExpConfig{}); err != nil {
			panic(err)
		}
	})

	cold, warm, gpuOut, err := timeGPUProfiled(iters, run)
	if err != nil {
		return benchResult{}, err
	}

	var want gnarkbn254.G1Affine
	if _, err := want.MultiExp(basesCPU, scalarsCPU, ecc.MultiExpConfig{}); err != nil {
		return benchResult{}, fmt.Errorf("cpu reference: %w", err)
	}

	return benchResult{
		cpu:    cpuResult,
		cold:   cold,
		warm:   warm,
		verify: len(gpuOut) == 1 && equalGPUToCPU(gpuOut[0], want),
		window: window,
	}, nil
}

func makeInputs(size int) ([]gnarkbn254.G1Affine, []gnarkfr.Element) {
	bases := make([]gnarkbn254.G1Affine, size)
	scalars := make([]gnarkfr.Element, size)
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

func pointsToGPU(in []gnarkbn254.G1Affine) []bn254.G1Affine {
	out := make([]bn254.G1Affine, len(in))
	for i := range in {
		out[i] = bn254.G1Affine{
			X: curvegpu.SplitWords4([4]uint64(in[i].X)),
			Y: curvegpu.SplitWords4([4]uint64(in[i].Y)),
		}
	}
	return out
}

func scalarsToGPU(in []gnarkfr.Element) []curvegpu.U32x8 {
	out := make([]curvegpu.U32x8, len(in))
	for i := range in {
		out[i] = scalarToRegularWords(in[i])
	}
	return out
}

func affineToGPUJac(in gnarkbn254.G1Affine) bn254.G1Jac {
	if in.IsInfinity() {
		return bn254.G1Jac{}
	}
	return bn254.G1Jac{
		X: curvegpu.SplitWords4([4]uint64(in.X)),
		Y: curvegpu.SplitWords4([4]uint64(in.Y)),
		Z: bn254.G1JacInfinity().X,
	}
}

func equalGPUToCPU(gpu bn254.G1Jac, want gnarkbn254.G1Affine) bool {
	gotJac := gnarkbn254.G1Jac{
		X: gnarkfp.Element(curvegpu.JoinWords8(gpu.X)),
		Y: gnarkfp.Element(curvegpu.JoinWords8(gpu.Y)),
		Z: gnarkfp.Element(curvegpu.JoinWords8(gpu.Z)),
	}
	var gotAff gnarkbn254.G1Affine
	gotAff.FromJacobian(&gotJac)
	return gotAff.Equal(&want)
}

func scalarToRegularWords(v gnarkfr.Element) curvegpu.U32x8 {
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

func timeCPU(iters int, fn func()) time.Duration {
	fn()
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn()
	}
	return time.Since(start) / time.Duration(iters)
}

func timeGPUProfiled(iters int, fn func() ([]bn254.G1Jac, bn254.G1MSMProfile, error)) (bn254.G1MSMProfile, bn254.G1MSMProfile, []bn254.G1Jac, error) {
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
	divisor := time.Duration(iters)
	warm.Partition /= divisor
	warm.ScalarMul /= divisor
	warm.BucketReduction /= divisor
	warm.WindowReduction /= divisor
	warm.FinalReduction /= divisor
	warm.Reduction /= divisor
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
