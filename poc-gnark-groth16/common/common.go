package common

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
)

type MulAddChainCircuit struct {
	X     frontend.Variable
	Y     frontend.Variable
	Out   frontend.Variable `gnark:",public"`
	Steps int               `gnark:"-"`
}

func (c *MulAddChainCircuit) Define(api frontend.API) error {
	acc := c.X
	for i := 0; i < c.Steps; i++ {
		acc = api.Add(api.Mul(acc, c.Y), 1)
	}
	api.AssertIsEqual(acc, c.Out)
	return nil
}

func ComputeOutput(field *big.Int, x uint64, y uint64, depth int) *big.Int {
	acc := new(big.Int).SetUint64(x)
	mul := new(big.Int).SetUint64(y)
	one := big.NewInt(1)

	for i := 0; i < depth; i++ {
		acc.Mul(acc, mul)
		acc.Mod(acc, field)
		acc.Add(acc, one)
		acc.Mod(acc, field)
	}

	return acc
}
