//go:build js && wasm

package groth16

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"sync"

	curve "github.com/consensys/gnark-crypto/ecc/bls12-377"
	bls12377fp "github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	bls12377fr "github.com/consensys/gnark-crypto/ecc/bls12-377/fr"
	"github.com/consensys/gnark/backend"
	native "github.com/consensys/gnark/backend/groth16/bls12-377"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bls12-377"
)

const (
	bls12377FrBytes           = 32
	bls12377G1CoordinateBytes = 48
	bls12377G1PointBytes      = 144
	bls12377G2ComponentBytes  = 48
	bls12377G2PointBytes      = 288
)

// BLS12377ProvingKey wraps gnark's native BLS12-377 Groth16 proving key with
// browser-side cached MSM bases.
type BLS12377ProvingKey struct {
	native.ProvingKey
	prepareMu      sync.Mutex
	scratchMu      sync.Mutex
	handle         string
	quotientWarmed bool
	g1AIndices     []int
	g1BIndices     []int
	scratch0       []byte
	scratch1       []byte
	scratch2       []byte
}

func proveBLS12377(r1cs *cs.R1CS, pk *BLS12377ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (*native.Proof, error) {
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("new prover config: %w", err)
	}

	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)
	if len(commitmentInfo) > 0 {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: commitment hints are not supported yet")
	}

	if err := pk.ensurePrepared(); err != nil {
		return nil, err
	}
	pk.scratchMu.Lock()
	defer pk.scratchMu.Unlock()

	_solution, err := r1cs.Solve(fullWitness, opt.SolverOpts...)
	if err != nil {
		return nil, err
	}
	solution := _solution.(*cs.R1CSSolution)
	wireValues := []bls12377fr.Element(solution.W)
	domainSize := int(pk.Domain.Cardinality)

	pk.scratch0 = packBLS12377FrVectorMontLEPaddedInto(pk.scratch0, solution.A, domainSize)
	pk.scratch1 = packBLS12377FrVectorMontLEPaddedInto(pk.scratch1, solution.B, domainSize)
	pk.scratch2 = packBLS12377FrVectorMontLEPaddedInto(pk.scratch2, solution.C, domainSize)
	zPacked, err := bridgeComputeHZMSMG1(pk.handle, pk.scratch0, pk.scratch1, pk.scratch2)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: quotient H + msm G1.Z: %w", err)
	}
	publicVariables := r1cs.GetNbPublicVariables()

	pk.scratch0, _ = packBLS12377FrVectorFilteredInto(pk.scratch0, wireValues, pk.g1AIndices, len(pk.InfinityA))
	pk.scratch1, _ = packBLS12377FrVectorFilteredInto(pk.scratch1, wireValues, pk.g1BIndices, len(pk.InfinityB))
	pk.scratch2 = packBLS12377FrVectorRegularLEInto(pk.scratch2, wireValues[publicVariables:])
	batchMSM, err := bridgeMSMBatch(pk.handle, pk.scratch0, pk.scratch1, pk.scratch2)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: batched MSMs: %w", err)
	}
	arBaseAff, err := decodeBLS12377G1AffineFromPacked(batchMSM.G1ABytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: msm G1.A: %w", err)
	}
	bs1BaseAff, err := decodeBLS12377G1AffineFromPacked(batchMSM.G1BBytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: msm G1.B: %w", err)
	}
	kBaseAff, err := decodeBLS12377G1AffineFromPacked(batchMSM.G1KBytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: msm G1.K: %w", err)
	}
	zBaseAff, err := decodeBLS12377G1AffineFromPacked(zPacked, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: msm G1.Z: %w", err)
	}
	bsBaseAff, err := decodeBLS12377G2AffineFromPacked(batchMSM.G2BBytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: msm G2.B: %w", err)
	}

	var r, s big.Int
	var _r, _s, _kr bls12377fr.Element
	if _, err := _r.SetRandom(); err != nil {
		return nil, err
	}
	if _, err := _s.SetRandom(); err != nil {
		return nil, err
	}
	_kr.Mul(&_r, &_s).Neg(&_kr)
	_r.BigInt(&r)
	_s.BigInt(&s)

	deltas := curve.BatchScalarMultiplicationG1(&pk.G1.Delta, []bls12377fr.Element{_r, _s, _kr})

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

func (pk *BLS12377ProvingKey) ensurePrepared() error {
	pk.prepareMu.Lock()
	defer pk.prepareMu.Unlock()

	if pk.handle != "" && pk.quotientWarmed {
		return nil
	}
	if err := bridgeInit("bls12_377"); err != nil {
		return err
	}

	if pk.handle == "" {
		payload := jsObject()
		payload.Set("g1A", jsUint8Array(packBLS12377G1AffineJacobianBatch(pk.G1.A)))
		payload.Set("g1ACount", len(pk.G1.A))
		payload.Set("g1B", jsUint8Array(packBLS12377G1AffineJacobianBatch(pk.G1.B)))
		payload.Set("g1BCount", len(pk.G1.B))
		payload.Set("g1K", jsUint8Array(packBLS12377G1AffineJacobianBatch(pk.G1.K)))
		payload.Set("g1KCount", len(pk.G1.K))
		payload.Set("g1Z", jsUint8Array(packBLS12377G1AffineJacobianBatch(pk.G1.Z)))
		payload.Set("g1ZCount", len(pk.G1.Z))
		payload.Set("g2B", jsUint8Array(packBLS12377G2AffineJacobianBatch(pk.G2.B)))
		payload.Set("g2BCount", len(pk.G2.B))

		handle, err := bridgePrepareKey("bls12_377", payload)
		if err != nil {
			return err
		}
		pk.handle = handle
		pk.g1AIndices = computeKeptIndices(pk.InfinityA)
		pk.g1BIndices = computeKeptIndices(pk.InfinityB)
	}
	if !pk.quotientWarmed {
		if err := bridgePrewarmQuotientDomain("bls12_377", int(pk.Domain.Cardinality)); err != nil {
			return err
		}
		pk.quotientWarmed = true
	}
	return nil
}

func packBLS12377FrVectorRegularLEInto(dst []byte, values []bls12377fr.Element) []byte {
	required := len(values) * bls12377FrBytes
	if cap(dst) < required {
		dst = make([]byte, required)
	} else {
		dst = dst[:required]
	}
	for i := range values {
		be := values[i].Bytes()
		base := i * bls12377FrBytes
		for j := 0; j < bls12377FrBytes; j++ {
			dst[base+j] = be[bls12377FrBytes-1-j]
		}
	}
	return dst
}

func packBLS12377FrVectorMontLEPaddedInto(dst []byte, values []bls12377fr.Element, size int) []byte {
	required := size * bls12377FrBytes
	if cap(dst) < required {
		dst = make([]byte, required)
	} else {
		dst = dst[:required]
		clear(dst)
	}
	for i := range values {
		base := i * bls12377FrBytes
		for j, word := range [4]uint64(values[i]) {
			binary.LittleEndian.PutUint64(dst[base+j*8:base+(j+1)*8], word)
		}
	}
	return dst
}

func packBLS12377FrVectorFilteredInto(dst []byte, values []bls12377fr.Element, keptPrefixIndices []int, prefixLen int) ([]byte, int) {
	limit := prefixLen
	if limit > len(values) {
		limit = len(values)
	}
	count := len(values) - limit
	for _, idx := range keptPrefixIndices {
		if idx >= limit {
			break
		}
		count++
	}
	required := count * bls12377FrBytes
	if cap(dst) < required {
		dst = make([]byte, required)
	} else {
		dst = dst[:required]
	}
	offset := 0
	for _, idx := range keptPrefixIndices {
		if idx >= limit {
			break
		}
		be := values[idx].Bytes()
		for j := 0; j < bls12377FrBytes; j++ {
			dst[offset+j] = be[bls12377FrBytes-1-j]
		}
		offset += bls12377FrBytes
	}
	for i := limit; i < len(values); i++ {
		be := values[i].Bytes()
		for j := 0; j < bls12377FrBytes; j++ {
			dst[offset+j] = be[bls12377FrBytes-1-j]
		}
		offset += bls12377FrBytes
	}
	return dst, count
}

func unpackBLS12377FrVectorRegularLE(packed []byte, err error) ([]bls12377fr.Element, error) {
	if err != nil {
		return nil, err
	}
	if len(packed)%bls12377FrBytes != 0 {
		return nil, fmt.Errorf("webgpu groth16 bls12_377: expected a multiple of %d fr bytes, got %d", bls12377FrBytes, len(packed))
	}
	count := len(packed) / bls12377FrBytes
	out := make([]bls12377fr.Element, count)
	var canonical [bls12377FrBytes]byte
	for i := 0; i < count; i++ {
		src := packed[i*bls12377FrBytes : (i+1)*bls12377FrBytes]
		for j := 0; j < bls12377FrBytes; j++ {
			canonical[bls12377FrBytes-1-j] = src[j]
		}
		out[i].SetBytes(canonical[:])
	}
	return out, nil
}

func packBLS12377G1AffineJacobianBatch(points []curve.G1Affine) []byte {
	out := make([]byte, 0, len(points)*bls12377G1PointBytes)
	one := bls12377FpOneMontLE()
	zero := make([]byte, bls12377G1CoordinateBytes)
	for i := range points {
		if points[i].IsInfinity() {
			out = append(out, zero...)
			out = append(out, zero...)
			out = append(out, zero...)
			continue
		}
		out = append(out, bls12377FpMontLE(points[i].X)...)
		out = append(out, bls12377FpMontLE(points[i].Y)...)
		out = append(out, one...)
	}
	return out
}

func packBLS12377G2AffineJacobianBatch(points []curve.G2Affine) []byte {
	out := make([]byte, 0, len(points)*bls12377G2PointBytes)
	one := bls12377FpOneMontLE()
	zero := make([]byte, bls12377G2ComponentBytes)
	for i := range points {
		if points[i].IsInfinity() {
			for j := 0; j < 6; j++ {
				out = append(out, zero...)
			}
			continue
		}
		out = append(out, bls12377FpMontLE(points[i].X.A0)...)
		out = append(out, bls12377FpMontLE(points[i].X.A1)...)
		out = append(out, bls12377FpMontLE(points[i].Y.A0)...)
		out = append(out, bls12377FpMontLE(points[i].Y.A1)...)
		out = append(out, one...)
		out = append(out, zero...)
	}
	return out
}

func decodeBLS12377G1AffineFromPacked(packed []byte, err error) (curve.G1Affine, error) {
	if err != nil {
		return curve.G1Affine{}, err
	}
	if len(packed) != 2*bls12377G1CoordinateBytes {
		return curve.G1Affine{}, fmt.Errorf("webgpu groth16 bls12_377: expected %d G1 bytes, got %d", 2*bls12377G1CoordinateBytes, len(packed))
	}
	return curve.G1Affine{
		X: readBLS12377FPMontLE(packed[:bls12377G1CoordinateBytes]),
		Y: readBLS12377FPMontLE(packed[bls12377G1CoordinateBytes:]),
	}, nil
}

func decodeBLS12377G2AffineFromPacked(packed []byte, err error) (curve.G2Affine, error) {
	if err != nil {
		return curve.G2Affine{}, err
	}
	if len(packed) != 4*bls12377G2ComponentBytes {
		return curve.G2Affine{}, fmt.Errorf("webgpu groth16 bls12_377: expected %d G2 bytes, got %d", 4*bls12377G2ComponentBytes, len(packed))
	}
	var out curve.G2Affine
	out.X.A0 = readBLS12377FPMontLE(packed[0*bls12377G2ComponentBytes : 1*bls12377G2ComponentBytes])
	out.X.A1 = readBLS12377FPMontLE(packed[1*bls12377G2ComponentBytes : 2*bls12377G2ComponentBytes])
	out.Y.A0 = readBLS12377FPMontLE(packed[2*bls12377G2ComponentBytes : 3*bls12377G2ComponentBytes])
	out.Y.A1 = readBLS12377FPMontLE(packed[3*bls12377G2ComponentBytes : 4*bls12377G2ComponentBytes])
	return out, nil
}

func readBLS12377FPMontLE(src []byte) bls12377fp.Element {
	var words [6]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(src[i*8 : (i+1)*8])
	}
	return bls12377fp.Element(words)
}

func bls12377FpMontLE(v bls12377fp.Element) []byte {
	out := make([]byte, bls12377G1CoordinateBytes)
	for i, word := range [6]uint64(v) {
		binary.LittleEndian.PutUint64(out[i*8:(i+1)*8], word)
	}
	return out
}

func bls12377FpOneMontLE() []byte {
	var one bls12377fp.Element
	one.SetOne()
	return bls12377FpMontLE(one)
}
