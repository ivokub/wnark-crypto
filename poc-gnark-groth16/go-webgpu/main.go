//go:build js && wasm

package main

import (
	"fmt"

	gnarkgroth16 "github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	webgpugroth16 "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16"
	"github.com/ivokub/wnark-crypto/poc-gnark-groth16/internal/pocutil"
)

func main() {
	if err := run(); err != nil {
		pocutil.Fail(err)
	}
}

func run() error {
	cfg, err := pocutil.ParseConfig()
	if err != nil {
		return err
	}

	return pocutil.RunHarness(
		fmt.Sprintf("Go Wasm -> WebGPU Groth16"),
		cfg,
		webgpugroth16.NewProvingKey,
		func(ccs constraint.ConstraintSystem, pk gnarkgroth16.ProvingKey, fullWitness witness.Witness) (gnarkgroth16.Proof, error) {
			return webgpugroth16.Prove(ccs, pk, fullWitness)
		},
		webgpugroth16.Prepare,
	)
}
