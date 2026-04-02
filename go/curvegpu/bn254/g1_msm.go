package bn254

import (
	"fmt"

	"github.com/gogpu/wgpu"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
)

type G1MSMKernel struct {
	g1 *G1Kernel
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
	if termsPerInstance <= 0 {
		return nil, fmt.Errorf("termsPerInstance must be positive")
	}
	if len(bases) != len(scalars) {
		return nil, fmt.Errorf("base/scalar length mismatch: %d != %d", len(bases), len(scalars))
	}
	if len(bases) == 0 {
		return nil, nil
	}
	if len(bases)%termsPerInstance != 0 {
		return nil, fmt.Errorf("base count %d is not divisible by termsPerInstance %d", len(bases), termsPerInstance)
	}

	count := len(bases) / termsPerInstance
	terms, err := k.g1.ScalarMulAffine(bases, scalars)
	if err != nil {
		return nil, fmt.Errorf("scalar multiply terms: %w", err)
	}
	width := termsPerInstance
	state := terms
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
			return nil, fmt.Errorf("reduce affine pairs: %w", err)
		}
		width = nextWidth
	}
	return state, nil
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
