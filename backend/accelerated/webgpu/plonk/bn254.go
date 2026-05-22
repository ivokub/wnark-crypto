//go:build js && wasm

package plonk

import (
	gnarkplonk "github.com/consensys/gnark/backend/plonk"
	native "github.com/consensys/gnark/backend/plonk/bn254"
	"github.com/consensys/gnark/backend/witness"
	cs "github.com/consensys/gnark/constraint/bn254"
)

// BN254ProvingKey wraps gnark's BN254 PLONK proving key. WebGPU-specific
// caches will hang off this type as we move prover phases out of wasm.
type BN254ProvingKey struct {
	native.ProvingKey
}

func (pk *BN254ProvingKey) ensurePrepared() error {
	// No WebGPU caches yet. Keep this method so the TS API can already expose
	// prepareProvingKey and the backend can grow without API churn.
	return nil
}

func proveBN254(spr *cs.SparseR1CS, pk *BN254ProvingKey, fullWitness witness.Witness) (gnarkplonk.Proof, error) {
	return native.Prove(spr, &pk.ProvingKey, fullWitness)
}
