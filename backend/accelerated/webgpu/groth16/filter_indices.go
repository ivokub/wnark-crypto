//go:build js && wasm

package groth16

func computeKeptIndices(infinity []bool) []int {
	if len(infinity) == 0 {
		return nil
	}
	count := 0
	for _, isInfinity := range infinity {
		if !isInfinity {
			count++
		}
	}
	indices := make([]int, 0, count)
	for i, isInfinity := range infinity {
		if !isInfinity {
			indices = append(indices, i)
		}
	}
	return indices
}
