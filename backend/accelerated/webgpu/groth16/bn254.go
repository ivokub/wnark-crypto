//go:build js && wasm

package groth16

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"sync"

	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	bn254fp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	bn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend"
	native "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
)

const (
	bn254FrBytes           = 32
	bn254G1CoordinateBytes = 32
	bn254G1PointBytes      = 96
	bn254G2ComponentBytes  = 32
	bn254G2PointBytes      = 192
)

// BN254ProvingKey wraps gnark's native BN254 Groth16 proving key with
// browser-side cached MSM bases.
type BN254ProvingKey struct {
	native.ProvingKey
	prepareMu sync.Mutex
	handle    string
}

func proveBN254(r1cs *cs.R1CS, pk *BN254ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (*native.Proof, error) {
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("new prover config: %w", err)
	}

	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)
	if len(commitmentInfo) > 0 {
		return nil, fmt.Errorf("webgpu groth16 bn254: commitment hints are not supported yet")
	}

	if err := pk.ensurePrepared(); err != nil {
		return nil, err
	}

	_solution, err := r1cs.Solve(fullWitness, opt.SolverOpts...)
	if err != nil {
		return nil, err
	}
	solution := _solution.(*cs.R1CSSolution)
	wireValues := []bn254fr.Element(solution.W)

	h, err := bridgeComputeHBN254(solution.A, solution.B, solution.C, int(pk.Domain.Cardinality))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: quotient H: %w", err)
	}
	wireValuesA := filterWireValuesBN254(wireValues, pk.InfinityA)
	wireValuesB := filterWireValuesBN254(wireValues, pk.InfinityB)
	privateWireValues := append([]bn254fr.Element(nil), wireValues[r1cs.GetNbPublicVariables():]...)

	arBaseAff, err := decodeBN254G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1A", packBN254FrVectorRegularLE(wireValuesA)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.A: %w", err)
	}
	bs1BaseAff, err := decodeBN254G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1B", packBN254FrVectorRegularLE(wireValuesB)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.B: %w", err)
	}
	kBaseAff, err := decodeBN254G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1K", packBN254FrVectorRegularLE(privateWireValues)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.K: %w", err)
	}
	sizeH := int(pk.Domain.Cardinality - 1)
	zBaseAff, err := decodeBN254G1AffineFromPacked(bridgeMSMG1(pk.handle, "g1Z", packBN254FrVectorRegularLE(h[:sizeH])))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.Z: %w", err)
	}
	bsBaseAff, err := decodeBN254G2AffineFromPacked(bridgeMSMG2(pk.handle, "g2B", packBN254FrVectorRegularLE(wireValuesB)))
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G2.B: %w", err)
	}

	var r, s big.Int
	var _r, _s, _kr bn254fr.Element
	if _, err := _r.SetRandom(); err != nil {
		return nil, err
	}
	if _, err := _s.SetRandom(); err != nil {
		return nil, err
	}
	_kr.Mul(&_r, &_s).Neg(&_kr)
	_r.BigInt(&r)
	_s.BigInt(&s)

	deltas := curve.BatchScalarMultiplicationG1(&pk.G1.Delta, []bn254fr.Element{_r, _s, _kr})

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

func (pk *BN254ProvingKey) ensurePrepared() error {
	pk.prepareMu.Lock()
	defer pk.prepareMu.Unlock()

	if pk.handle != "" {
		return nil
	}
	if err := bridgeInit("bn254"); err != nil {
		return err
	}

	payload := jsObject()
	payload.Set("g1A", jsUint8Array(packBN254G1AffineJacobianBatch(pk.G1.A)))
	payload.Set("g1ACount", len(pk.G1.A))
	payload.Set("g1B", jsUint8Array(packBN254G1AffineJacobianBatch(pk.G1.B)))
	payload.Set("g1BCount", len(pk.G1.B))
	payload.Set("g1K", jsUint8Array(packBN254G1AffineJacobianBatch(pk.G1.K)))
	payload.Set("g1KCount", len(pk.G1.K))
	payload.Set("g1Z", jsUint8Array(packBN254G1AffineJacobianBatch(pk.G1.Z)))
	payload.Set("g1ZCount", len(pk.G1.Z))
	payload.Set("g2B", jsUint8Array(packBN254G2AffineJacobianBatch(pk.G2.B)))
	payload.Set("g2BCount", len(pk.G2.B))

	handle, err := bridgePrepareKey("bn254", payload)
	if err != nil {
		return err
	}
	pk.handle = handle
	return nil
}

func filterWireValuesBN254(values []bn254fr.Element, infinity []bool) []bn254fr.Element {
	if len(infinity) == 0 {
		return append([]bn254fr.Element(nil), values...)
	}
	out := make([]bn254fr.Element, 0, len(values))
	for i := range values {
		if i < len(infinity) && infinity[i] {
			continue
		}
		out = append(out, values[i])
	}
	return out
}

func bridgeComputeHBN254(a, b, c []bn254fr.Element, domainSize int) ([]bn254fr.Element, error) {
	aPacked := packBN254FrVectorRegularLE(padBN254FrVector(a, domainSize))
	bPacked := packBN254FrVectorRegularLE(padBN254FrVector(b, domainSize))
	cPacked := packBN254FrVectorRegularLE(padBN254FrVector(c, domainSize))
	return unpackBN254FrVectorRegularLE(bridgeComputeH("bn254", aPacked, bPacked, cPacked))
}

func packBN254FrVectorRegularLE(values []bn254fr.Element) []byte {
	out := make([]byte, 0, len(values)*bn254FrBytes)
	for i := range values {
		be := values[i].Bytes()
		out = append(out, reverseBytes(be[:])...)
	}
	return out
}

func padBN254FrVector(values []bn254fr.Element, size int) []bn254fr.Element {
	out := make([]bn254fr.Element, size)
	copy(out, values)
	return out
}

func unpackBN254FrVectorRegularLE(packed []byte, err error) ([]bn254fr.Element, error) {
	if err != nil {
		return nil, err
	}
	if len(packed)%bn254FrBytes != 0 {
		return nil, fmt.Errorf("webgpu groth16 bn254: expected a multiple of %d fr bytes, got %d", bn254FrBytes, len(packed))
	}
	count := len(packed) / bn254FrBytes
	out := make([]bn254fr.Element, count)
	var canonical [bn254FrBytes]byte
	for i := 0; i < count; i++ {
		src := packed[i*bn254FrBytes : (i+1)*bn254FrBytes]
		for j := 0; j < bn254FrBytes; j++ {
			canonical[bn254FrBytes-1-j] = src[j]
		}
		out[i].SetBytes(canonical[:])
	}
	return out, nil
}

func packBN254G1AffineJacobianBatch(points []curve.G1Affine) []byte {
	out := make([]byte, 0, len(points)*bn254G1PointBytes)
	one := bn254FpOneMontLE()
	zero := make([]byte, bn254G1CoordinateBytes)
	for i := range points {
		if points[i].IsInfinity() {
			out = append(out, zero...)
			out = append(out, zero...)
			out = append(out, zero...)
			continue
		}
		out = append(out, bn254FpMontLE(points[i].X)...)
		out = append(out, bn254FpMontLE(points[i].Y)...)
		out = append(out, one...)
	}
	return out
}

func packBN254G2AffineJacobianBatch(points []curve.G2Affine) []byte {
	out := make([]byte, 0, len(points)*bn254G2PointBytes)
	one := bn254FpOneMontLE()
	zero := make([]byte, bn254G2ComponentBytes)
	for i := range points {
		if points[i].IsInfinity() {
			for j := 0; j < 6; j++ {
				out = append(out, zero...)
			}
			continue
		}
		out = append(out, bn254FpMontLE(points[i].X.A0)...)
		out = append(out, bn254FpMontLE(points[i].X.A1)...)
		out = append(out, bn254FpMontLE(points[i].Y.A0)...)
		out = append(out, bn254FpMontLE(points[i].Y.A1)...)
		out = append(out, one...)
		out = append(out, zero...)
	}
	return out
}

func decodeBN254G1AffineFromPacked(packed []byte, err error) (curve.G1Affine, error) {
	if err != nil {
		return curve.G1Affine{}, err
	}
	if len(packed) != 2*bn254G1CoordinateBytes {
		return curve.G1Affine{}, fmt.Errorf("webgpu groth16 bn254: expected %d G1 bytes, got %d", 2*bn254G1CoordinateBytes, len(packed))
	}
	return curve.G1Affine{
		X: readBN254FPMontLE(packed[:bn254G1CoordinateBytes]),
		Y: readBN254FPMontLE(packed[bn254G1CoordinateBytes:]),
	}, nil
}

func decodeBN254G2AffineFromPacked(packed []byte, err error) (curve.G2Affine, error) {
	if err != nil {
		return curve.G2Affine{}, err
	}
	if len(packed) != 4*bn254G2ComponentBytes {
		return curve.G2Affine{}, fmt.Errorf("webgpu groth16 bn254: expected %d G2 bytes, got %d", 4*bn254G2ComponentBytes, len(packed))
	}
	var out curve.G2Affine
	out.X.A0 = readBN254FPMontLE(packed[0*bn254G2ComponentBytes : 1*bn254G2ComponentBytes])
	out.X.A1 = readBN254FPMontLE(packed[1*bn254G2ComponentBytes : 2*bn254G2ComponentBytes])
	out.Y.A0 = readBN254FPMontLE(packed[2*bn254G2ComponentBytes : 3*bn254G2ComponentBytes])
	out.Y.A1 = readBN254FPMontLE(packed[3*bn254G2ComponentBytes : 4*bn254G2ComponentBytes])
	return out, nil
}

func readBN254FPMontLE(src []byte) bn254fp.Element {
	var words [4]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(src[i*8 : (i+1)*8])
	}
	return bn254fp.Element(words)
}

func bn254FpMontLE(v bn254fp.Element) []byte {
	out := make([]byte, bn254G1CoordinateBytes)
	for i, word := range [4]uint64(v) {
		binary.LittleEndian.PutUint64(out[i*8:(i+1)*8], word)
	}
	return out
}

func bn254FpOneMontLE() []byte {
	var one bn254fp.Element
	one.SetOne()
	return bn254FpMontLE(one)
}

func reverseBytes(src []byte) []byte {
	out := make([]byte, len(src))
	for i := range src {
		out[i] = src[len(src)-1-i]
	}
	return out
}
