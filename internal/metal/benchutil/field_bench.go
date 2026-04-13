package benchutil

import (
	"time"

	"github.com/ivokub/wnark-crypto/internal/metal"
)

type VectorBenchCase[P any] struct {
	Name     string
	CPU      func()
	GPU      func() ([]metal.U32x8, P, error)
	Expected []metal.U32x8
}

type VectorBenchResult[P any] struct {
	Name   string
	CPU    time.Duration
	Cold   P
	Warm   P
	Verify bool
}

type NTTBenchCase[P any] struct {
	Name     string
	Init     time.Duration
	CPU      func()
	GPU      func() ([]metal.U32x8, P, error)
	Expected []metal.U32x8
}

type NTTBenchResult[P any] struct {
	Name   string
	Init   time.Duration
	CPU    time.Duration
	Cold   P
	Warm   P
	Verify bool
}

func RunVectorBenchmarks[P any](
	iters int,
	cases []VectorBenchCase[P],
	zero func() P,
	add func(*P, P),
	div func(*P, int),
) ([]VectorBenchResult[P], error) {
	results := make([]VectorBenchResult[P], 0, len(cases))
	for _, c := range cases {
		cpu := TimeCPU(iters, c.CPU)
		cold, warm, out, err := AverageProfiled(iters, c.GPU, zero, add, div)
		if err != nil {
			return nil, err
		}
		results = append(results, VectorBenchResult[P]{
			Name:   c.Name,
			CPU:    cpu,
			Cold:   cold,
			Warm:   warm,
			Verify: EqualGPUBatches(out, c.Expected),
		})
	}
	return results, nil
}

func RunNTTBenchmarks[P any](
	iters int,
	cases []NTTBenchCase[P],
	zero func() P,
	add func(*P, P),
	div func(*P, int),
) ([]NTTBenchResult[P], error) {
	results := make([]NTTBenchResult[P], 0, len(cases))
	for _, c := range cases {
		cpu := TimeCPU(iters, c.CPU)
		cold, warm, out, err := AverageProfiled(iters, c.GPU, zero, add, div)
		if err != nil {
			return nil, err
		}
		results = append(results, NTTBenchResult[P]{
			Name:   c.Name,
			Init:   c.Init,
			CPU:    cpu,
			Cold:   cold,
			Warm:   warm,
			Verify: EqualGPUBatches(out, c.Expected),
		})
	}
	return results, nil
}

func EqualGPUBatches(a, b []metal.U32x8) bool {
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
