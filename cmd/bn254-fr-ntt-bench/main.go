package main

import (
	"flag"
	"fmt"
	mrand "math/rand"
	"strings"
	"time"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
	gnarkfft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/utils"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	"github.com/ivokub/wnark-crypto/go/curvegpu/bn254"
)

type benchResult struct {
	name   string
	init   time.Duration
	cpu    time.Duration
	cold   bn254.FrNTTProfile
	warm   bn254.FrNTTProfile
	verify bool
}

func main() {
	var (
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

	fmt.Println("=== BN254 fr NTT Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", *minLog, *maxLog)
	fmt.Printf("iterations: %d\n\n", *iters)

	fmt.Println("size,op,init_ms,cpu_ms,cold_bit_reverse_ms,cold_stage_upload_ms,cold_stage_kernel_ms,cold_stage_readback_ms,cold_stage_total_ms,cold_scale_ms,cold_total_ms,cold_with_init_ms,warm_bit_reverse_ms,warm_stage_total_ms,warm_scale_ms,warm_total_ms,verified")
	for logSize := *minLog; logSize <= *maxLog; logSize++ {
		size := 1 << logSize
		results, err := runSize(size, *iters)
		if err != nil {
			if isResourceError(err) {
				fmt.Printf("# stop at size=2^%d: %v\n", logSize, err)
				break
			}
			panic(err)
		}
		for _, result := range results {
			fmt.Printf(
				"%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%t\n",
				size,
				result.name,
				durationMS(result.init),
				durationMS(result.cpu),
				durationMS(result.cold.BitReverse.Total),
				durationMS(result.cold.Stages.Upload),
				durationMS(result.cold.Stages.Kernel),
				durationMS(result.cold.Stages.Readback),
				durationMS(result.cold.Stages.Total),
				durationMS(result.cold.Scale.Total),
				durationMS(result.cold.Total),
				durationMS(result.init+result.cold.Total),
				durationMS(result.warm.BitReverse.Total),
				durationMS(result.warm.Stages.Total),
				durationMS(result.warm.Scale.Total),
				durationMS(result.warm.Total),
				result.verify,
			)
		}
	}
}

func runSize(size, iters int) ([]benchResult, error) {
	domain := gnarkfft.NewDomain(uint64(size))
	stageTwiddles, inverseStageTwiddles, scale, err := buildTwiddles(domain, size)
	if err != nil {
		return nil, err
	}

	input := makeInput(size)
	forwardExpected := cpuForward(domain, input)
	inverseExpected := cpuInverse(domain, forwardExpected)

	inputGPU := vectorToGPU(input)
	forwardExpectedGPU := vectorToGPU(forwardExpected)

	forwardKernel, forwardInit, err := newNTTKernel()
	if err != nil {
		return nil, err
	}
	defer forwardKernel.close()

	inverseKernel, inverseInit, err := newNTTKernel()
	if err != nil {
		return nil, err
	}
	defer inverseKernel.close()

	return runBenchmarks(forwardKernel, inverseKernel, forwardInit, inverseInit, domain, input, forwardExpected, inverseExpected, inputGPU, forwardExpectedGPU, stageTwiddles, inverseStageTwiddles, scale, iters)
}

func runBenchmarks(
	forwardKernel kernelSet,
	inverseKernel kernelSet,
	forwardInit time.Duration,
	inverseInit time.Duration,
	domain *gnarkfft.Domain,
	input, forwardExpected, inverseExpected gnarkfr.Vector,
	inputGPU, forwardExpectedGPU []curvegpu.U32x8,
	stageTwiddles, inverseStageTwiddles [][]curvegpu.U32x8,
	scale curvegpu.U32x8,
	iters int,
) ([]benchResult, error) {
	results := make([]benchResult, 0, 2)

	cpuForwardMs := timeCPU(iters, func() {
		_ = cpuForward(domain, input)
	})
	coldForward, warmForward, forwardOut, err := timeNTTProfiled(iters, func() ([]curvegpu.U32x8, bn254.FrNTTProfile, error) {
		return forwardKernel.kernel.ForwardDITFullPathProfiled(inputGPU, stageTwiddles)
	})
	if err != nil {
		return nil, fmt.Errorf("bench forward_ntt: %w", err)
	}
	results = append(results, benchResult{
		name:   "forward_ntt",
		init:   forwardInit,
		cpu:    cpuForwardMs,
		cold:   coldForward,
		warm:   warmForward,
		verify: equalGPUBatches(forwardOut, vectorToGPU(forwardExpected)),
	})

	cpuInverseMs := timeCPU(iters, func() {
		_ = cpuInverse(domain, forwardExpected)
	})
	coldInverse, warmInverse, inverseOut, err := timeNTTProfiled(iters, func() ([]curvegpu.U32x8, bn254.FrNTTProfile, error) {
		return inverseKernel.kernel.InverseDITFullPathProfiled(forwardExpectedGPU, inverseStageTwiddles, scale)
	})
	if err != nil {
		return nil, fmt.Errorf("bench inverse_ntt: %w", err)
	}
	results = append(results, benchResult{
		name:   "inverse_ntt",
		init:   inverseInit,
		cpu:    cpuInverseMs,
		cold:   coldInverse,
		warm:   warmInverse,
		verify: equalGPUBatches(inverseOut, vectorToGPU(inverseExpected)),
	})

	return results, nil
}

type kernelSet struct {
	device *bn254.DeviceSet
	kernel *bn254.FrNTTKernel
}

func (k kernelSet) close() {
	if k.kernel != nil {
		k.kernel.Close()
	}
	if k.device != nil {
		k.device.Close()
	}
}

func newNTTKernel() (kernelSet, time.Duration, error) {
	start := time.Now()
	deviceSet, err := bn254.NewHeadlessDevice()
	if err != nil {
		return kernelSet{}, 0, err
	}
	kernel, err := bn254.NewFrNTTKernel(deviceSet.Device)
	if err != nil {
		deviceSet.Close()
		return kernelSet{}, 0, err
	}
	return kernelSet{device: deviceSet, kernel: kernel}, time.Since(start), nil
}

func buildTwiddles(domain *gnarkfft.Domain, size int) ([][]curvegpu.U32x8, [][]curvegpu.U32x8, curvegpu.U32x8, error) {
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
		stageTwiddles[stage-1] = vectorToGPU(gnarkfr.Vector(twiddles[logN-stage][:m]))
		inverseStageTwiddles[stage-1] = vectorToGPU(gnarkfr.Vector(twiddlesInv[logN-stage][:m]))
	}
	return stageTwiddles, inverseStageTwiddles, curvegpu.SplitWords4([4]uint64(domain.CardinalityInv)), nil
}

func makeInput(size int) gnarkfr.Vector {
	out := make(gnarkfr.Vector, size)
	rng := mrand.New(mrand.NewSource(int64(size)*2654435761 + 7))
	for i := range out {
		out[i].SetUint64(rng.Uint64())
	}
	return out
}

func cpuForward(domain *gnarkfft.Domain, input gnarkfr.Vector) gnarkfr.Vector {
	state := make(gnarkfr.Vector, len(input))
	copy(state, input)
	utils.BitReverse(state)
	domain.FFT(state, gnarkfft.DIT)
	return state
}

func cpuInverse(domain *gnarkfft.Domain, input gnarkfr.Vector) gnarkfr.Vector {
	state := make(gnarkfr.Vector, len(input))
	copy(state, input)
	utils.BitReverse(state)
	domain.FFTInverse(state, gnarkfft.DIT)
	return state
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

func timeNTTProfiled(iters int, fn func() ([]curvegpu.U32x8, bn254.FrNTTProfile, error)) (bn254.FrNTTProfile, bn254.FrNTTProfile, []curvegpu.U32x8, error) {
	last, cold, err := fn()
	if err != nil {
		return bn254.FrNTTProfile{}, bn254.FrNTTProfile{}, nil, err
	}
	if iters == 1 {
		return cold, cold, last, nil
	}
	var warm bn254.FrNTTProfile
	for i := 0; i < iters; i++ {
		var profile bn254.FrNTTProfile
		last, profile, err = fn()
		if err != nil {
			return bn254.FrNTTProfile{}, bn254.FrNTTProfile{}, nil, err
		}
		warm.BitReverse.Upload += profile.BitReverse.Upload
		warm.BitReverse.Kernel += profile.BitReverse.Kernel
		warm.BitReverse.Readback += profile.BitReverse.Readback
		warm.BitReverse.Total += profile.BitReverse.Total
		warm.Stages.Upload += profile.Stages.Upload
		warm.Stages.Kernel += profile.Stages.Kernel
		warm.Stages.Readback += profile.Stages.Readback
		warm.Stages.Total += profile.Stages.Total
		warm.Scale.Upload += profile.Scale.Upload
		warm.Scale.Kernel += profile.Scale.Kernel
		warm.Scale.Readback += profile.Scale.Readback
		warm.Scale.Total += profile.Scale.Total
		warm.Total += profile.Total
	}
	divisor := time.Duration(iters)
	warm.BitReverse.Upload /= divisor
	warm.BitReverse.Kernel /= divisor
	warm.BitReverse.Readback /= divisor
	warm.BitReverse.Total /= divisor
	warm.Stages.Upload /= divisor
	warm.Stages.Kernel /= divisor
	warm.Stages.Readback /= divisor
	warm.Stages.Total /= divisor
	warm.Scale.Upload /= divisor
	warm.Scale.Kernel /= divisor
	warm.Scale.Readback /= divisor
	warm.Scale.Total /= divisor
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
