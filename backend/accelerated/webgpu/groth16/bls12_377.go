//go:build js && wasm

package groth16

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"strconv"
	"sync"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bls12-377"
	bls12377fp "github.com/consensys/gnark-crypto/ecc/bls12-377/fp"
	bls12377fr "github.com/consensys/gnark-crypto/ecc/bls12-377/fr"
	"github.com/consensys/gnark-crypto/ecc/bls12-377/fr/hash_to_field"
	"github.com/consensys/gnark/backend"
	native "github.com/consensys/gnark/backend/groth16/bls12-377"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bls12-377"
	"github.com/consensys/gnark/constraint/solver"
	fcs "github.com/consensys/gnark/frontend/cs"
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
	if opt.HashToFieldFn == nil {
		opt.HashToFieldFn = hash_to_field.New([]byte(constraint.CommitmentDst))
	}

	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)

	if err := pk.ensurePrepared(); err != nil {
		return nil, err
	}
	pk.scratchMu.Lock()
	defer pk.scratchMu.Unlock()

	proof := &native.Proof{
		Commitments: make([]curve.G1Affine, len(commitmentInfo)),
	}
	privateCommittedValues := make([][]bls12377fr.Element, len(commitmentInfo))
	solverOpts := opt.SolverOpts[:len(opt.SolverOpts):len(opt.SolverOpts)]
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	solverOpts = append(solverOpts, solver.OverrideHint(bsb22ID, func(_ *big.Int, in []*big.Int, out []*big.Int) error {
		i := int(in[0].Int64())
		if i < 0 || i >= len(commitmentInfo) {
			return fmt.Errorf("webgpu groth16 bls12_377: invalid commitment index %d", i)
		}
		in = in[1:]
		hashedCount := len(commitmentInfo[i].PublicAndCommitmentCommitted)
		if len(in) < hashedCount {
			return fmt.Errorf("webgpu groth16 bls12_377: commitment hint %d has %d inputs, expected at least %d", i, len(in), hashedCount)
		}
		hashed := in[:hashedCount]
		committed := in[hashedCount:]

		privateCommittedValues[i] = make([]bls12377fr.Element, len(committed))
		for j, inJ := range committed {
			privateCommittedValues[i][j].SetBigInt(inJ)
		}

		scalars := packBLS12377FrVectorRegularLEInto(nil, privateCommittedValues[i])
		commitmentPacked, err := bridgeMSMG1(pk.handle, "commitmentBasis"+strconv.Itoa(i), scalars)
		if err != nil {
			return fmt.Errorf("webgpu groth16 bls12_377: commitment %d MSM: %w", i, err)
		}
		if proof.Commitments[i], err = decodeBLS12377G1AffineFromPacked(commitmentPacked, nil); err != nil {
			return fmt.Errorf("webgpu groth16 bls12_377: commitment %d decode: %w", i, err)
		}

		if _, err := opt.HashToFieldFn.Write(constraint.SerializeCommitment(proof.Commitments[i].Marshal(), hashed, (bls12377fr.Bits-1)/8+1)); err != nil {
			return err
		}
		hashBts := opt.HashToFieldFn.Sum(nil)
		opt.HashToFieldFn.Reset()
		nbBuf := bls12377fr.Bytes
		if opt.HashToFieldFn.Size() < bls12377fr.Bytes {
			nbBuf = opt.HashToFieldFn.Size()
		}
		var res bls12377fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(out[0])
		return nil
	}))

	_solution, err := r1cs.Solve(fullWitness, solverOpts...)
	if err != nil {
		return nil, err
	}
	solution := _solution.(*cs.R1CSSolution)
	wireValues := []bls12377fr.Element(solution.W)
	domainSize := int(pk.Domain.Cardinality)

	if len(commitmentInfo) > 0 {
		poks := make([]curve.G1Affine, len(commitmentInfo))
		for i := range commitmentInfo {
			if privateCommittedValues[i] == nil {
				return nil, fmt.Errorf("webgpu groth16 bls12_377: commitment hint %d was not evaluated", i)
			}
			scalars := packBLS12377FrVectorRegularLEInto(nil, privateCommittedValues[i])
			pokPacked, err := bridgeMSMG1(pk.handle, "commitmentBasisExpSigma"+strconv.Itoa(i), scalars)
			if err != nil {
				return nil, fmt.Errorf("webgpu groth16 bls12_377: commitment %d pok MSM: %w", i, err)
			}
			if poks[i], err = decodeBLS12377G1AffineFromPacked(pokPacked, nil); err != nil {
				return nil, fmt.Errorf("webgpu groth16 bls12_377: commitment %d pok decode: %w", i, err)
			}
		}
		commitmentsSerialized := make([]byte, bls12377fr.Bytes*len(commitmentInfo))
		for i := range commitmentInfo {
			copy(commitmentsSerialized[bls12377fr.Bytes*i:], wireValues[commitmentInfo[i].CommitmentIndex].Marshal())
		}
		challenge, err := bls12377fr.Hash(commitmentsSerialized, []byte("G16-BSB22"), 1)
		if err != nil {
			return nil, err
		}
		if _, err = proof.CommitmentPok.Fold(poks, challenge[0], ecc.MultiExpConfig{NbTasks: 1}); err != nil {
			return nil, err
		}
	}

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
	pk.scratch2 = packBLS12377FrVectorRegularLEFilteredOutInto(pk.scratch2, wireValues[publicVariables:], publicVariables, commitmentWireIndexesToRemove(commitmentInfo))
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
		payload.Set("commitmentCount", len(pk.CommitmentKeys))
		for i := range pk.CommitmentKeys {
			suffix := strconv.Itoa(i)
			payload.Set("commitmentBasis"+suffix, jsUint8Array(packBLS12377G1AffineJacobianBatch(pk.CommitmentKeys[i].Basis)))
			payload.Set("commitmentBasis"+suffix+"Count", len(pk.CommitmentKeys[i].Basis))
			payload.Set("commitmentBasisExpSigma"+suffix, jsUint8Array(packBLS12377G1AffineJacobianBatch(pk.CommitmentKeys[i].BasisExpSigma)))
			payload.Set("commitmentBasisExpSigma"+suffix+"Count", len(pk.CommitmentKeys[i].BasisExpSigma))
		}

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
		base := i * bls12377FrBytes
		writeBLS12377FrRegularLE(dst[base:base+bls12377FrBytes], &values[i])
	}
	return dst
}

func packBLS12377FrVectorRegularLEFilteredOutInto(dst []byte, values []bls12377fr.Element, firstIndex int, remove []int) []byte {
	if len(remove) == 0 {
		return packBLS12377FrVectorRegularLEInto(dst, values)
	}
	removeSet := make(map[int]struct{}, len(remove))
	for _, idx := range remove {
		removeSet[idx] = struct{}{}
	}
	count := 0
	for i := range values {
		if _, ok := removeSet[firstIndex+i]; !ok {
			count++
		}
	}
	required := count * bls12377FrBytes
	if cap(dst) < required {
		dst = make([]byte, required)
	} else {
		dst = dst[:required]
	}
	offset := 0
	for i := range values {
		if _, ok := removeSet[firstIndex+i]; ok {
			continue
		}
		writeBLS12377FrRegularLE(dst[offset:offset+bls12377FrBytes], &values[i])
		offset += bls12377FrBytes
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
		writeBLS12377FrMontLE(dst[base:base+bls12377FrBytes], &values[i])
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
		writeBLS12377FrRegularLE(dst[offset:offset+bls12377FrBytes], &values[idx])
		offset += bls12377FrBytes
	}
	for i := limit; i < len(values); i++ {
		writeBLS12377FrRegularLE(dst[offset:offset+bls12377FrBytes], &values[i])
		offset += bls12377FrBytes
	}
	return dst, count
}

func writeBLS12377FrRegularLE(dst []byte, value *bls12377fr.Element) {
	be := value.Bytes()
	for i := 0; i < bls12377FrBytes; i++ {
		dst[i] = be[bls12377FrBytes-1-i]
	}
}

func writeBLS12377FrMontLE(dst []byte, value *bls12377fr.Element) {
	for i, word := range [4]uint64(*value) {
		binary.LittleEndian.PutUint64(dst[i*8:(i+1)*8], word)
	}
}

func packBLS12377G1AffineJacobianBatch(points []curve.G1Affine) []byte {
	out := make([]byte, len(points)*bls12377G1PointBytes)
	one := bls12377FpOneMontLE()
	for i := range points {
		if points[i].IsInfinity() {
			continue
		}
		base := i * bls12377G1PointBytes
		writeBLS12377FPMontLE(out[base:base+bls12377G1CoordinateBytes], &points[i].X)
		writeBLS12377FPMontLE(out[base+bls12377G1CoordinateBytes:base+2*bls12377G1CoordinateBytes], &points[i].Y)
		copy(out[base+2*bls12377G1CoordinateBytes:base+3*bls12377G1CoordinateBytes], one)
	}
	return out
}

func packBLS12377G2AffineJacobianBatch(points []curve.G2Affine) []byte {
	out := make([]byte, len(points)*bls12377G2PointBytes)
	one := bls12377FpOneMontLE()
	for i := range points {
		if points[i].IsInfinity() {
			continue
		}
		base := i * bls12377G2PointBytes
		writeBLS12377FPMontLE(out[base:base+bls12377G2ComponentBytes], &points[i].X.A0)
		writeBLS12377FPMontLE(out[base+bls12377G2ComponentBytes:base+2*bls12377G2ComponentBytes], &points[i].X.A1)
		writeBLS12377FPMontLE(out[base+2*bls12377G2ComponentBytes:base+3*bls12377G2ComponentBytes], &points[i].Y.A0)
		writeBLS12377FPMontLE(out[base+3*bls12377G2ComponentBytes:base+4*bls12377G2ComponentBytes], &points[i].Y.A1)
		copy(out[base+4*bls12377G2ComponentBytes:base+5*bls12377G2ComponentBytes], one)
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

func writeBLS12377FPMontLE(dst []byte, value *bls12377fp.Element) {
	for i, word := range [6]uint64(*value) {
		binary.LittleEndian.PutUint64(dst[i*8:(i+1)*8], word)
	}
}

func bls12377FpOneMontLE() []byte {
	out := make([]byte, bls12377G1CoordinateBytes)
	var one bls12377fp.Element
	one.SetOne()
	writeBLS12377FPMontLE(out, &one)
	return out
}
