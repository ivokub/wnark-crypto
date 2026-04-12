package main

import (
	"flag"
	"fmt"
	mrand "math/rand"
	"time"

	gnarkfr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkfft "github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	"github.com/consensys/gnark-crypto/utils"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
	bls12381 "github.com/ivokub/wnark-crypto/go/curvegpu/bls12_381"
	"github.com/ivokub/wnark-crypto/internal/benchutil"
)

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

	fmt.Println("=== BLS12-381 fr NTT Performance Benchmark ===")
	fmt.Printf("sizes: 2^%d .. 2^%d\n", *minLog, *maxLog)
	fmt.Printf("iterations: %d\n\n", *iters)

	fmt.Println("size,op,init_ms,cpu_ms,cold_bit_reverse_ms,cold_stage_upload_ms,cold_stage_kernel_ms,cold_stage_readback_ms,cold_stage_total_ms,cold_scale_ms,cold_total_ms,cold_with_init_ms,warm_bit_reverse_ms,warm_stage_total_ms,warm_scale_ms,warm_total_ms,verified")
	for logSize := *minLog; logSize <= *maxLog; logSize++ {
		size := 1 << logSize
		results, err := runSize(size, *iters)
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

func runSize(size, iters int) ([]benchutil.NTTBenchResult[bls12381.FrNTTProfile], error) {
	domain := gnarkfft.NewDomain(uint64(size))
	stageTwiddles, inverseStageTwiddles, scale, err := buildTwiddles(domain)
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
) ([]benchutil.NTTBenchResult[bls12381.FrNTTProfile], error) {
	return benchutil.RunNTTBenchmarks(iters, []benchutil.NTTBenchCase[bls12381.FrNTTProfile]{
		{
			Name: "forward_ntt",
			Init: forwardInit,
			CPU:  func() { _ = cpuForward(domain, input) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrNTTProfile, error) {
				out, profile, err := forwardKernel.kernel.ForwardDITFullPathProfiled(inputGPU, stageTwiddles)
				if err != nil {
					return nil, bls12381.FrNTTProfile{}, fmt.Errorf("bench forward_ntt: %w", err)
				}
				return out, profile, nil
			},
			Expected: vectorToGPU(forwardExpected),
		},
		{
			Name: "inverse_ntt",
			Init: inverseInit,
			CPU:  func() { _ = cpuInverse(domain, forwardExpected) },
			GPU: func() ([]curvegpu.U32x8, bls12381.FrNTTProfile, error) {
				out, profile, err := inverseKernel.kernel.InverseDITFullPathProfiled(forwardExpectedGPU, inverseStageTwiddles, scale)
				if err != nil {
					return nil, bls12381.FrNTTProfile{}, fmt.Errorf("bench inverse_ntt: %w", err)
				}
				return out, profile, nil
			},
			Expected: vectorToGPU(inverseExpected),
		},
	}, zeroFrNTTProfile, addFrNTTProfile, divFrNTTProfile)
}

type kernelSet struct {
	device *bls12381.DeviceSet
	kernel *bls12381.FrNTTKernel
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
	deviceSet, err := bls12381.NewHeadlessDevice()
	if err != nil {
		return kernelSet{}, 0, err
	}
	kernel, err := bls12381.NewFrNTTKernel(deviceSet.Device)
	if err != nil {
		deviceSet.Close()
		return kernelSet{}, 0, err
	}
	return kernelSet{device: deviceSet, kernel: kernel}, time.Since(start), nil
}

func buildTwiddles(domain *gnarkfft.Domain) ([][]curvegpu.U32x8, [][]curvegpu.U32x8, curvegpu.U32x8, error) {
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

func zeroFrNTTProfile() bls12381.FrNTTProfile {
	return bls12381.FrNTTProfile{}
}

func addFrNTTProfile(dst *bls12381.FrNTTProfile, src bls12381.FrNTTProfile) {
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

func divFrNTTProfile(dst *bls12381.FrNTTProfile, divisor int) {
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
