//go:build js && wasm

package groth16

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"sync"

	curve "github.com/consensys/gnark-crypto/ecc/bls12-381"
	bls12381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	bls12381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	"github.com/consensys/gnark/backend"
	native "github.com/consensys/gnark/backend/groth16/bls12-381"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bls12-381"
)

const (
	bls12381FrBytes           = 32
	bls12381G1CoordinateBytes = 48
	bls12381G1PointBytes      = 144
	bls12381G2ComponentBytes  = 48
	bls12381G2PointBytes      = 288
)

// BLS12381ProvingKey wraps gnark's native BLS12-381 Groth16 proving key with
// browser-side cached MSM bases.
type BLS12381ProvingKey struct {
	native.ProvingKey
	prepareMu sync.Mutex
	handle    string
}

func proveBLS12381(r1cs *cs.R1CS, pk *BLS12381ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (*native.Proof, error) {
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("new prover config: %w", err)
	}

	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)
	if len(commitmentInfo) > 0 {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: commitment hints are not supported yet")
	}

	if err := pk.ensurePrepared(); err != nil {
		return nil, err
	}

	_solution, err := r1cs.Solve(fullWitness, opt.SolverOpts...)
	if err != nil {
		return nil, err
	}
	solution := _solution.(*cs.R1CSSolution)
	wireValues := []bls12381fr.Element(solution.W)

	h, err := bridgeComputeHBLS12381(solution.A, solution.B, solution.C, int(pk.Domain.Cardinality))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: quotient H: %w", err)
	}
	wireValuesA := filterWireValuesBLS12381(wireValues, pk.InfinityA)
	wireValuesB := filterWireValuesBLS12381(wireValues, pk.InfinityB)
	privateWireValues := append([]bls12381fr.Element(nil), wireValues[r1cs.GetNbPublicVariables():]...)

	arBaseAff, err := decodeBLS12381G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1A", packBLS12381FrVectorRegularLE(wireValuesA)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: msm G1.A: %w", err)
	}
	bs1BaseAff, err := decodeBLS12381G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1B", packBLS12381FrVectorRegularLE(wireValuesB)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: msm G1.B: %w", err)
	}
	kBaseAff, err := decodeBLS12381G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1K", packBLS12381FrVectorRegularLE(privateWireValues)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: msm G1.K: %w", err)
	}
	sizeH := int(pk.Domain.Cardinality - 1)
	zBaseAff, err := decodeBLS12381G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1Z", packBLS12381FrVectorRegularLE(h[:sizeH])))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: msm G1.Z: %w", err)
	}
	bsBaseAff, err := decodeBLS12381G2AffineFromPacked(bridgeMSMG2(pk.handle, "g2B", packBLS12381FrVectorRegularLE(wireValuesB)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: msm G2.B: %w", err)
	}

	var r, s big.Int
	var _r, _s, _kr bls12381fr.Element
	if _, err := _r.SetRandom(); err != nil {
		return nil, err
	}
	if _, err := _s.SetRandom(); err != nil {
		return nil, err
	}
	_kr.Mul(&_r, &_s).Neg(&_kr)
	_r.BigInt(&r)
	_s.BigInt(&s)

	deltas := curve.BatchScalarMultiplicationG1(&pk.G1.Delta, []bls12381fr.Element{_r, _s, _kr})

	var ar, bs1, krs, krs2, tmp curve.G1Jac
	ar.FromAffine(&arBaseAff)
	ar.AddMixed(&pk.G1.Alpha)
	ar.AddMixed(&deltas[0])

	bs1.FromAffine(&bs1BaseAff)
	bs1.AddMixed(&pk.G1.Beta)
	bs1.AddMixed(&deltas[1])

	krs.FromAffine(&kBaseAff)
	krs2.FromAffine(&zBaseAff)
	krs.AddAssign(&krs2)
	krs.AddMixed(&deltas[2])

	tmp.ScalarMultiplication(&ar, &s)
	krs.AddAssign(&tmp)
	tmp.ScalarMultiplication(&bs1, &r)
	krs.AddAssign(&tmp)

	var bs, deltaS curve.G2Jac
	bs.FromAffine(&bsBaseAff)
	deltaS.FromAffine(&pk.G2.Delta)
	deltaS.ScalarMultiplication(&deltaS, &s)
	bs.AddAssign(&deltaS)
	bs.AddMixed(&pk.G2.Beta)

	proof := &native.Proof{
		Commitments: make([]curve.G1Affine, 0),
	}
	proof.Ar.FromJacobian(&ar)
	proof.Krs.FromJacobian(&krs)
	proof.Bs.FromJacobian(&bs)
	return proof, nil
}

func (pk *BLS12381ProvingKey) ensurePrepared() error {
	pk.prepareMu.Lock()
	defer pk.prepareMu.Unlock()

	if pk.handle != "" {
		return nil
	}
	if err := bridgeInit("bls12_381"); err != nil {
		return err
	}

	payload := jsObject()
	payload.Set("g1A", jsUint8Array(packBLS12381G1AffineJacobianBatch(pk.G1.A)))
	payload.Set("g1ACount", len(pk.G1.A))
	payload.Set("g1B", jsUint8Array(packBLS12381G1AffineJacobianBatch(pk.G1.B)))
	payload.Set("g1BCount", len(pk.G1.B))
	payload.Set("g1K", jsUint8Array(packBLS12381G1AffineJacobianBatch(pk.G1.K)))
	payload.Set("g1KCount", len(pk.G1.K))
	payload.Set("g1Z", jsUint8Array(packBLS12381G1AffineJacobianBatch(pk.G1.Z)))
	payload.Set("g1ZCount", len(pk.G1.Z))
	payload.Set("g2B", jsUint8Array(packBLS12381G2AffineJacobianBatch(pk.G2.B)))
	payload.Set("g2BCount", len(pk.G2.B))

	handle, err := bridgePrepareKey("bls12_381", payload)
	if err != nil {
		return err
	}
	pk.handle = handle
	return nil
}

func filterWireValuesBLS12381(values []bls12381fr.Element, infinity []bool) []bls12381fr.Element {
	if len(infinity) == 0 {
		return append([]bls12381fr.Element(nil), values...)
	}
	out := make([]bls12381fr.Element, 0, len(values))
	for i := range values {
		if i < len(infinity) && infinity[i] {
			continue
		}
		out = append(out, values[i])
	}
	return out
}

func bridgeComputeHBLS12381(a, b, c []bls12381fr.Element, domainSize int) ([]bls12381fr.Element, error) {
	aPacked := packBLS12381FrVectorRegularLE(padBLS12381FrVector(a, domainSize))
	bPacked := packBLS12381FrVectorRegularLE(padBLS12381FrVector(b, domainSize))
	cPacked := packBLS12381FrVectorRegularLE(padBLS12381FrVector(c, domainSize))
	return unpackBLS12381FrVectorRegularLE(bridgeComputeH("bls12_381", aPacked, bPacked, cPacked))
}

func packBLS12381FrVectorRegularLE(values []bls12381fr.Element) []byte {
	out := make([]byte, 0, len(values)*bls12381FrBytes)
	for i := range values {
		be := values[i].Bytes()
		out = append(out, reverseBytes(be[:])...)
	}
	return out
}

func padBLS12381FrVector(values []bls12381fr.Element, size int) []bls12381fr.Element {
	out := make([]bls12381fr.Element, size)
	copy(out, values)
	return out
}

func unpackBLS12381FrVectorRegularLE(packed []byte, err error) ([]bls12381fr.Element, error) {
	if err != nil {
		return nil, err
	}
	if len(packed)%bls12381FrBytes != 0 {
		return nil, fmt.Errorf("webgpu groth16 bls12_381: expected a multiple of %d fr bytes, got %d", bls12381FrBytes, len(packed))
	}
	count := len(packed) / bls12381FrBytes
	out := make([]bls12381fr.Element, count)
	var canonical [bls12381FrBytes]byte
	for i := 0; i < count; i++ {
		src := packed[i*bls12381FrBytes : (i+1)*bls12381FrBytes]
		for j := 0; j < bls12381FrBytes; j++ {
			canonical[bls12381FrBytes-1-j] = src[j]
		}
		out[i].SetBytes(canonical[:])
	}
	return out, nil
}

func packBLS12381G1AffineJacobianBatch(points []curve.G1Affine) []byte {
	out := make([]byte, 0, len(points)*bls12381G1PointBytes)
	one := bls12381FpOneMontLE()
	zero := make([]byte, bls12381G1CoordinateBytes)
	for i := range points {
		if points[i].IsInfinity() {
			out = append(out, zero...)
			out = append(out, zero...)
			out = append(out, zero...)
			continue
		}
		out = append(out, bls12381FpMontLE(points[i].X)...)
		out = append(out, bls12381FpMontLE(points[i].Y)...)
		out = append(out, one...)
	}
	return out
}

func packBLS12381G2AffineJacobianBatch(points []curve.G2Affine) []byte {
	out := make([]byte, 0, len(points)*bls12381G2PointBytes)
	one := bls12381FpOneMontLE()
	zero := make([]byte, bls12381G2ComponentBytes)
	for i := range points {
		if points[i].IsInfinity() {
			for j := 0; j < 6; j++ {
				out = append(out, zero...)
			}
			continue
		}
		out = append(out, bls12381FpMontLE(points[i].X.A0)...)
		out = append(out, bls12381FpMontLE(points[i].X.A1)...)
		out = append(out, bls12381FpMontLE(points[i].Y.A0)...)
		out = append(out, bls12381FpMontLE(points[i].Y.A1)...)
		out = append(out, one...)
		out = append(out, zero...)
	}
	return out
}

func decodeBLS12381G1AffineFromPacked(packed []byte, err error) (curve.G1Affine, error) {
	if err != nil {
		return curve.G1Affine{}, err
	}
	if len(packed) != 2*bls12381G1CoordinateBytes {
		return curve.G1Affine{}, fmt.Errorf("webgpu groth16 bls12_381: expected %d G1 bytes, got %d", 2*bls12381G1CoordinateBytes, len(packed))
	}
	return curve.G1Affine{
		X: readBLS12381FPMontLE(packed[:bls12381G1CoordinateBytes]),
		Y: readBLS12381FPMontLE(packed[bls12381G1CoordinateBytes:]),
	}, nil
}

func decodeBLS12381G2AffineFromPacked(packed []byte, err error) (curve.G2Affine, error) {
	if err != nil {
		return curve.G2Affine{}, err
	}
	if len(packed) != 4*bls12381G2ComponentBytes {
		return curve.G2Affine{}, fmt.Errorf("webgpu groth16 bls12_381: expected %d G2 bytes, got %d", 4*bls12381G2ComponentBytes, len(packed))
	}
	var out curve.G2Affine
	out.X.A0 = readBLS12381FPMontLE(packed[0*bls12381G2ComponentBytes : 1*bls12381G2ComponentBytes])
	out.X.A1 = readBLS12381FPMontLE(packed[1*bls12381G2ComponentBytes : 2*bls12381G2ComponentBytes])
	out.Y.A0 = readBLS12381FPMontLE(packed[2*bls12381G2ComponentBytes : 3*bls12381G2ComponentBytes])
	out.Y.A1 = readBLS12381FPMontLE(packed[3*bls12381G2ComponentBytes : 4*bls12381G2ComponentBytes])
	return out, nil
}

func readBLS12381FPMontLE(src []byte) bls12381fp.Element {
	var words [6]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(src[i*8 : (i+1)*8])
	}
	return bls12381fp.Element(words)
}

func bls12381FpMontLE(v bls12381fp.Element) []byte {
	out := make([]byte, bls12381G1CoordinateBytes)
	for i, word := range [6]uint64(v) {
		binary.LittleEndian.PutUint64(out[i*8:(i+1)*8], word)
	}
	return out
}

func bls12381FpOneMontLE() []byte {
	var one bls12381fp.Element
	one.SetOne()
	return bls12381FpMontLE(one)
}
