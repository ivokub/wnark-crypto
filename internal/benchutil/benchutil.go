package benchutil

import (
	"fmt"
	"math/bits"
	"strings"
	"time"
)

func DurationMS(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

func IsResourceError(err error) bool {
	if err == nil {
		return false
	}
	text := strings.ToLower(err.Error())
	return strings.Contains(text, "buffer") ||
		strings.Contains(text, "resource") ||
		strings.Contains(text, "memory")
}

func TimeCPU(iters int, fn func()) time.Duration {
	fn()
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn()
	}
	return time.Since(start) / time.Duration(iters)
}

func BitReverseLogCount(count int) (uint32, error) {
	if count <= 0 {
		return 0, fmt.Errorf("bit reverse requires non-zero input")
	}
	if count&(count-1) != 0 {
		return 0, fmt.Errorf("bit reverse requires power-of-two length, got %d", count)
	}
	return uint32(bits.Len(uint(count)) - 1), nil
}

func AverageProfiled[T any, P any](
	iters int,
	fn func() (T, P, error),
	zero func() P,
	add func(*P, P),
	div func(*P, int),
) (P, P, T, error) {
	last, cold, err := fn()
	if err != nil {
		return zero(), zero(), last, err
	}
	if iters == 1 {
		return cold, cold, last, nil
	}
	warm := zero()
	for i := 0; i < iters; i++ {
		var profile P
		last, profile, err = fn()
		if err != nil {
			return zero(), zero(), last, err
		}
		add(&warm, profile)
	}
	div(&warm, iters)
	return cold, warm, last, nil
}
