//go:build js && wasm

package main

import (
	"fmt"

	gnarkgroth16 "github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
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
		fmt.Sprintf("Go Wasm -> Native Groth16"),
		cfg,
		gnarkgroth16.NewProvingKey,
		func(ccs constraint.ConstraintSystem, pk gnarkgroth16.ProvingKey, fullWitness witness.Witness) (gnarkgroth16.Proof, error) {
			return gnarkgroth16.Prove(ccs, pk, fullWitness)
		},
		nil,
	)
}
