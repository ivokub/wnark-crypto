package bn254

import (
	"fmt"
	"time"

	"github.com/gogpu/wgpu"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
)

type G1MSMKernel struct {
	g1 *G1Kernel
}

type G1MSMProfile struct {
	Partition       time.Duration
	ScalarMul       time.Duration
	BucketReduction time.Duration
	WindowReduction time.Duration
	FinalReduction  time.Duration
	Reduction       time.Duration
	Total           time.Duration
}

func NewG1MSMKernel(device *wgpu.Device) (*G1MSMKernel, error) {
	g1, err := NewG1Kernel(device)
	if err != nil {
		return nil, err
	}
	return &G1MSMKernel{
		g1: g1,
	}, nil
}

func (k *G1MSMKernel) Close() {
	if k == nil || k.g1 == nil {
		return
	}
	k.g1.Close()
}

func (k *G1MSMKernel) RunAffineNaive(bases []G1Affine, scalars []curvegpu.U32x8, termsPerInstance int) ([]G1Jac, error) {
	out, _, err := k.RunAffineNaiveProfiled(bases, scalars, termsPerInstance)
	return out, err
}

func (k *G1MSMKernel) RunAffineNaiveProfiled(bases []G1Affine, scalars []curvegpu.U32x8, termsPerInstance int) ([]G1Jac, G1MSMProfile, error) {
	if termsPerInstance <= 0 {
		return nil, G1MSMProfile{}, fmt.Errorf("termsPerInstance must be positive")
	}
	if len(bases) != len(scalars) {
		return nil, G1MSMProfile{}, fmt.Errorf("base/scalar length mismatch: %d != %d", len(bases), len(scalars))
	}
	if len(bases) == 0 {
		return nil, G1MSMProfile{}, nil
	}
	if len(bases)%termsPerInstance != 0 {
		return nil, G1MSMProfile{}, fmt.Errorf("base count %d is not divisible by termsPerInstance %d", len(bases), termsPerInstance)
	}

	totalStart := time.Now()
	count := len(bases) / termsPerInstance
	scalarMulStart := time.Now()
	terms, err := k.g1.ScalarMulAffine(bases, scalars)
	if err != nil {
		return nil, G1MSMProfile{}, fmt.Errorf("scalar multiply terms: %w", err)
	}
	profile := G1MSMProfile{
		ScalarMul: time.Since(scalarMulStart),
	}

	width := termsPerInstance
	state := terms
	reductionStart := time.Now()
	for width > 1 {
		nextWidth := (width + 1) / 2
		left := make([]G1Affine, count*nextWidth)
		right := make([]G1Affine, count*nextWidth)
		for instance := 0; instance < count; instance++ {
			rowBase := instance * width
			nextBase := instance * nextWidth
			for j := 0; j < nextWidth; j++ {
				left[nextBase+j] = jacAffineToAffine(state[rowBase+2*j])
				if 2*j+1 < width {
					right[nextBase+j] = jacAffineToAffine(state[rowBase+2*j+1])
				}
			}
		}
		state, err = k.g1.AffineAdd(left, right)
		if err != nil {
			return nil, G1MSMProfile{}, fmt.Errorf("reduce affine pairs: %w", err)
		}
		width = nextWidth
	}
	profile.Reduction = time.Since(reductionStart)
	profile.Total = time.Since(totalStart)
	return state, profile, nil
}

func (k *G1MSMKernel) RunAffinePippenger(bases []G1Affine, scalars []curvegpu.U32x8, termsPerInstance int, window uint32) ([]G1Jac, error) {
	out, _, err := k.RunAffinePippengerProfiled(bases, scalars, termsPerInstance, window)
	return out, err
}

func BestPippengerWindow(nbPoints int) uint32 {
	if nbPoints <= 0 {
		return 4
	}
	implementedWindows := []uint32{4, 5, 6, 7, 8, 9, 10, 11, 12}
	best := implementedWindows[0]
	minCost := int(^uint(0) >> 1)
	for _, window := range implementedWindows {
		cost := ((255 + int(window)) / int(window)) * (nbPoints + (1 << window))
		if cost < minCost {
			minCost = cost
			best = window
		}
	}
	return best
}

func (k *G1MSMKernel) RunAffinePippengerProfiled(bases []G1Affine, scalars []curvegpu.U32x8, termsPerInstance int, window uint32) ([]G1Jac, G1MSMProfile, error) {
	if termsPerInstance <= 0 {
		return nil, G1MSMProfile{}, fmt.Errorf("termsPerInstance must be positive")
	}
	if len(bases) != len(scalars) {
		return nil, G1MSMProfile{}, fmt.Errorf("base/scalar length mismatch: %d != %d", len(bases), len(scalars))
	}
	if len(bases) == 0 {
		return nil, G1MSMProfile{}, nil
	}
	if len(bases)%termsPerInstance != 0 {
		return nil, G1MSMProfile{}, fmt.Errorf("base count %d is not divisible by termsPerInstance %d", len(bases), termsPerInstance)
	}
	if window == 0 || window > 16 {
		return nil, G1MSMProfile{}, fmt.Errorf("window must be in [1,16], got %d", window)
	}

	totalStart := time.Now()
	count := len(bases) / termsPerInstance
	numWindows := int((256 + window - 1) / window)
	bucketCount := (1 << window) - 1
	profile := G1MSMProfile{}

	partitionStart := time.Now()
	bucketLists := make([][]G1Affine, count*numWindows*bucketCount)
	for instance := 0; instance < count; instance++ {
		baseOffset := instance * termsPerInstance
		for term := 0; term < termsPerInstance; term++ {
			idx := baseOffset + term
			scalar := scalars[idx]
			if scalar == (curvegpu.U32x8{}) {
				continue
			}
			for win := 0; win < numWindows; win++ {
				digit := extractWindowDigit(scalar, uint32(win)*window, window)
				if digit == 0 {
					continue
				}
				slot := ((instance*numWindows)+win)*bucketCount + int(digit-1)
				bucketLists[slot] = append(bucketLists[slot], bases[idx])
			}
		}
	}
	profile.Partition = time.Since(partitionStart)

	bucketReduceStart := time.Now()
	bucketSums, err := k.reduceAffineBuckets(bucketLists)
	if err != nil {
		return nil, G1MSMProfile{}, fmt.Errorf("reduce buckets: %w", err)
	}
	profile.BucketReduction = time.Since(bucketReduceStart)

	windowReduceStart := time.Now()
	windowSums, err := k.reduceWindows(bucketSums, count, numWindows, bucketCount)
	if err != nil {
		return nil, G1MSMProfile{}, fmt.Errorf("reduce windows: %w", err)
	}
	profile.WindowReduction = time.Since(windowReduceStart)

	finalReduceStart := time.Now()
	out, err := k.combineWindows(windowSums, count, numWindows, window)
	if err != nil {
		return nil, G1MSMProfile{}, fmt.Errorf("combine windows: %w", err)
	}
	profile.FinalReduction = time.Since(finalReduceStart)
	profile.Reduction = profile.BucketReduction + profile.WindowReduction + profile.FinalReduction
	profile.Total = time.Since(totalStart)
	return out, profile, nil
}

func (k *G1MSMKernel) reduceAffineBuckets(bucketLists [][]G1Affine) ([]G1Affine, error) {
	current := make([][]G1Affine, len(bucketLists))
	copy(current, bucketLists)
	for {
		next := make([][]G1Affine, len(current))
		left := make([]G1Affine, 0)
		right := make([]G1Affine, 0)
		type mapEntry struct {
			bucket int
			slot   int
		}
		mappings := make([]mapEntry, 0)
		work := false
		for bucketIdx, bucket := range current {
			switch len(bucket) {
			case 0:
			case 1:
				next[bucketIdx] = bucket
			default:
				work = true
				nextCount := (len(bucket) + 1) / 2
				next[bucketIdx] = make([]G1Affine, nextCount)
				for j := 0; j < nextCount; j++ {
					left = append(left, bucket[2*j])
					if 2*j+1 < len(bucket) {
						right = append(right, bucket[2*j+1])
					} else {
						right = append(right, G1Affine{})
					}
					mappings = append(mappings, mapEntry{bucket: bucketIdx, slot: j})
				}
			}
		}
		if !work {
			out := make([]G1Affine, len(current))
			for i, bucket := range current {
				if len(bucket) == 1 {
					out[i] = bucket[0]
				}
			}
			return out, nil
		}
		reduced, err := k.affineAddSafe(left, right)
		if err != nil {
			return nil, err
		}
		for i, point := range reduced {
			entry := mappings[i]
			next[entry.bucket][entry.slot] = jacAffineToAffine(point)
		}
		current = next
	}
}

func (k *G1MSMKernel) reduceWindows(bucketSums []G1Affine, count, numWindows, bucketCount int) ([]G1Jac, error) {
	totalSlots := count * numWindows
	running := make([]G1Jac, totalSlots)
	windowTotals := make([]G1Jac, totalSlots)
	for i := range running {
		running[i] = G1JacInfinity()
		windowTotals[i] = G1JacInfinity()
	}

	for bucket := bucketCount - 1; bucket >= 0; bucket-- {
		activeRunIdx := make([]int, 0)
		activeRunJac := make([]G1Jac, 0)
		activeRunAff := make([]G1Affine, 0)
		for slot := 0; slot < totalSlots; slot++ {
			bucketPoint := bucketSums[slot*bucketCount+bucket]
			if bucketPoint.IsInfinity() {
				continue
			}
			activeRunIdx = append(activeRunIdx, slot)
			activeRunJac = append(activeRunJac, running[slot])
			activeRunAff = append(activeRunAff, bucketPoint)
		}
		if len(activeRunIdx) > 0 {
			out, err := k.addMixedSafe(activeRunJac, activeRunAff)
			if err != nil {
				return nil, err
			}
			for i, slot := range activeRunIdx {
				running[slot] = out[i]
			}
		}

		activeTotalIdx := make([]int, 0)
		activeTotalJac := make([]G1Jac, 0)
		activeRunningJac := make([]G1Jac, 0)
		for slot := 0; slot < totalSlots; slot++ {
			if running[slot].IsInfinity() {
				continue
			}
			activeTotalIdx = append(activeTotalIdx, slot)
			activeTotalJac = append(activeTotalJac, windowTotals[slot])
			activeRunningJac = append(activeRunningJac, running[slot])
		}
		if len(activeTotalIdx) == 0 {
			continue
		}
		runningAffJac, err := k.jacToAffineSafe(activeRunningJac)
		if err != nil {
			return nil, err
		}
		runningAff := make([]G1Affine, len(runningAffJac))
		for i, point := range runningAffJac {
			runningAff[i] = jacAffineToAffine(point)
		}
		out, err := k.addMixedSafe(activeTotalJac, runningAff)
		if err != nil {
			return nil, err
		}
		for i, slot := range activeTotalIdx {
			windowTotals[slot] = out[i]
		}
	}
	return windowTotals, nil
}

func (k *G1MSMKernel) combineWindows(windowSums []G1Jac, count, numWindows int, window uint32) ([]G1Jac, error) {
	windowAff := make([]G1Affine, len(windowSums))
	activeWindowIdx := make([]int, 0)
	activeWindowJac := make([]G1Jac, 0)
	for i, point := range windowSums {
		if point.IsInfinity() {
			continue
		}
		activeWindowIdx = append(activeWindowIdx, i)
		activeWindowJac = append(activeWindowJac, point)
	}
	if len(activeWindowIdx) > 0 {
		affJac, err := k.jacToAffineSafe(activeWindowJac)
		if err != nil {
			return nil, err
		}
		for i, idx := range activeWindowIdx {
			windowAff[idx] = jacAffineToAffine(affJac[i])
		}
	}

	acc := make([]G1Jac, count)
	for i := range acc {
		acc[i] = G1JacInfinity()
	}

	for win := numWindows - 1; win >= 0; win-- {
		if win != numWindows-1 {
			for step := uint32(0); step < window; step++ {
				doubled, err := k.doubleJacSafe(acc)
				if err != nil {
					return nil, err
				}
				acc = doubled
			}
		}

		activeIdx := make([]int, 0)
		activeJac := make([]G1Jac, 0)
		activeAff := make([]G1Affine, 0)
		for instance := 0; instance < count; instance++ {
			point := windowAff[instance*numWindows+win]
			if point.IsInfinity() {
				continue
			}
			activeIdx = append(activeIdx, instance)
			activeJac = append(activeJac, acc[instance])
			activeAff = append(activeAff, point)
		}
		if len(activeIdx) == 0 {
			continue
		}
		out, err := k.addMixedSafe(activeJac, activeAff)
		if err != nil {
			return nil, err
		}
		for i, instance := range activeIdx {
			acc[instance] = out[i]
		}
	}

	return acc, nil
}

func extractWindowDigit(scalar curvegpu.U32x8, bitOffset, window uint32) uint32 {
	if window == 0 {
		return 0
	}
	word := bitOffset / 32
	shift := bitOffset % 32
	mask := uint32((1 << window) - 1)
	if word >= 8 {
		return 0
	}
	if shift+window <= 32 {
		return (scalar[word] >> shift) & mask
	}
	lowBits := scalar[word] >> shift
	if word+1 >= 8 {
		return lowBits & mask
	}
	highWidth := shift + window - 32
	highMask := uint32((1 << highWidth) - 1)
	highBits := scalar[word+1] & highMask
	return (lowBits | (highBits << (32 - shift))) & mask
}

func jacAffineToAffine(in G1Jac) G1Affine {
	if in.IsInfinity() {
		return G1Affine{}
	}
	return G1Affine{
		X: in.X,
		Y: in.Y,
	}
}

func (k *G1MSMKernel) affineAddSafe(a, b []G1Affine) ([]G1Jac, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("affineAddSafe length mismatch: %d != %d", len(a), len(b))
	}
	if len(a) == 0 {
		return nil, nil
	}
	if len(a) == 1 {
		out, err := k.g1.AffineAdd(
			[]G1Affine{a[0], {}},
			[]G1Affine{b[0], {}},
		)
		if err != nil {
			return nil, err
		}
		return out[:1], nil
	}
	return k.g1.AffineAdd(a, b)
}

func (k *G1MSMKernel) addMixedSafe(p []G1Jac, q []G1Affine) ([]G1Jac, error) {
	if len(p) != len(q) {
		return nil, fmt.Errorf("addMixedSafe length mismatch: %d != %d", len(p), len(q))
	}
	if len(p) == 0 {
		return nil, nil
	}
	if len(p) == 1 {
		out, err := k.g1.AddMixed(
			[]G1Jac{p[0], G1JacInfinity()},
			[]G1Affine{q[0], {}},
		)
		if err != nil {
			return nil, err
		}
		return out[:1], nil
	}
	return k.g1.AddMixed(p, q)
}

func (k *G1MSMKernel) doubleJacSafe(p []G1Jac) ([]G1Jac, error) {
	if len(p) == 0 {
		return nil, nil
	}
	if len(p) == 1 {
		out, err := k.g1.DoubleJac([]G1Jac{p[0], G1JacInfinity()})
		if err != nil {
			return nil, err
		}
		return out[:1], nil
	}
	return k.g1.DoubleJac(p)
}

func (k *G1MSMKernel) jacToAffineSafe(p []G1Jac) ([]G1Jac, error) {
	if len(p) == 0 {
		return nil, nil
	}
	if len(p) == 1 {
		out, err := k.g1.JacToAffine([]G1Jac{p[0], G1JacInfinity()})
		if err != nil {
			return nil, err
		}
		return out[:1], nil
	}
	return k.g1.JacToAffine(p)
}
