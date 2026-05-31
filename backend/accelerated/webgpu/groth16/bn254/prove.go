//go:build js && wasm

package bn254

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"strconv"
	"sync"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	"github.com/consensys/gnark/backend"
	native "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	fcs "github.com/consensys/gnark/frontend/cs"
	"github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16/internal/bridge"
	"github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/groth16/internal/common"
)

const (
	frBytes           = 32
	g1CoordinateBytes = 32
	g1PointBytes      = 96
	g2ComponentBytes  = 32
	g2PointBytes      = 192
)

// ProvingKey wraps gnark's native BN254 Groth16 proving key with
// browser-side cached MSM bases.
type ProvingKey struct {
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

func Prove(r1cs *cs.R1CS, pk *ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (*native.Proof, error) {
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("new prover config: %w", err)
	}
	if opt.HashToFieldFn == nil {
		opt.HashToFieldFn = hash_to_field.New([]byte(constraint.CommitmentDst))
	}

	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)

	if err := pk.Prepare(); err != nil {
		return nil, err
	}
	pk.scratchMu.Lock()
	defer pk.scratchMu.Unlock()

	proof := &native.Proof{
		Commitments: make([]bn254.G1Affine, len(commitmentInfo)),
	}
	privateCommittedValues := make([][]fr.Element, len(commitmentInfo))
	solverOpts := opt.SolverOpts[:len(opt.SolverOpts):len(opt.SolverOpts)]
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	solverOpts = append(solverOpts, solver.OverrideHint(bsb22ID, func(_ *big.Int, in []*big.Int, out []*big.Int) error {
		i := int(in[0].Int64())
		if i < 0 || i >= len(commitmentInfo) {
			return fmt.Errorf("webgpu groth16 bn254: invalid commitment index %d", i)
		}
		in = in[1:]
		hashedCount := len(commitmentInfo[i].PublicAndCommitmentCommitted)
		if len(in) < hashedCount {
			return fmt.Errorf("webgpu groth16 bn254: commitment hint %d has %d inputs, expected at least %d", i, len(in), hashedCount)
		}
		hashed := in[:hashedCount]
		committed := in[hashedCount:]

		privateCommittedValues[i] = make([]fr.Element, len(committed))
		for j, inJ := range committed {
			privateCommittedValues[i][j].SetBigInt(inJ)
		}

		scalars := packFrVectorRegularLEInto(nil, privateCommittedValues[i])
		commitmentPacked, err := bridge.Bridge.MSMG1(pk.handle, "commitmentBasis"+strconv.Itoa(i), scalars)
		if err != nil {
			return fmt.Errorf("webgpu groth16 bn254: commitment %d MSM: %w", i, err)
		}
		if proof.Commitments[i], err = decodeG1AffineFromPacked(commitmentPacked, nil); err != nil {
			return fmt.Errorf("webgpu groth16 bn254: commitment %d decode: %w", i, err)
		}

		if _, err := opt.HashToFieldFn.Write(constraint.SerializeCommitment(proof.Commitments[i].Marshal(), hashed, (fr.Bits-1)/8+1)); err != nil {
			return err
		}
		hashBts := opt.HashToFieldFn.Sum(nil)
		opt.HashToFieldFn.Reset()
		nbBuf := fr.Bytes
		if opt.HashToFieldFn.Size() < fr.Bytes {
			nbBuf = opt.HashToFieldFn.Size()
		}
		var res fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(out[0])
		return nil
	}))

	_solution, err := r1cs.Solve(fullWitness, solverOpts...)
	if err != nil {
		return nil, err
	}
	solution := _solution.(*cs.R1CSSolution)
	wireValues := []fr.Element(solution.W)
	domainSize := int(pk.Domain.Cardinality)

	if len(commitmentInfo) > 0 {
		poks := make([]bn254.G1Affine, len(commitmentInfo))
		for i := range commitmentInfo {
			if privateCommittedValues[i] == nil {
				return nil, fmt.Errorf("webgpu groth16 bn254: commitment hint %d was not evaluated", i)
			}
			scalars := packFrVectorRegularLEInto(nil, privateCommittedValues[i])
			pokPacked, err := bridge.Bridge.MSMG1(pk.handle, "commitmentBasisExpSigma"+strconv.Itoa(i), scalars)
			if err != nil {
				return nil, fmt.Errorf("webgpu groth16 bn254: commitment %d pok MSM: %w", i, err)
			}
			if poks[i], err = decodeG1AffineFromPacked(pokPacked, nil); err != nil {
				return nil, fmt.Errorf("webgpu groth16 bn254: commitment %d pok decode: %w", i, err)
			}
		}
		commitmentsSerialized := make([]byte, fr.Bytes*len(commitmentInfo))
		for i := range commitmentInfo {
			copy(commitmentsSerialized[fr.Bytes*i:], wireValues[commitmentInfo[i].CommitmentIndex].Marshal())
		}
		challenge, err := fr.Hash(commitmentsSerialized, []byte("G16-BSB22"), 1)
		if err != nil {
			return nil, err
		}
		if _, err = proof.CommitmentPok.Fold(poks, challenge[0], ecc.MultiExpConfig{NbTasks: 1}); err != nil {
			return nil, err
		}
	}

	pk.scratch0 = packFrVectorMontLEPaddedInto(pk.scratch0, solution.A, domainSize)
	pk.scratch1 = packFrVectorMontLEPaddedInto(pk.scratch1, solution.B, domainSize)
	pk.scratch2 = packFrVectorMontLEPaddedInto(pk.scratch2, solution.C, domainSize)
	zPacked, err := bridge.Bridge.ComputeHZMSMG1(pk.handle, pk.scratch0, pk.scratch1, pk.scratch2)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: quotient H + msm G1.Z: %w", err)
	}
	publicVariables := r1cs.GetNbPublicVariables()

	pk.scratch0, _ = packFrVectorFilteredInto(pk.scratch0, wireValues, pk.g1AIndices, len(pk.InfinityA))
	pk.scratch1, _ = packFrVectorFilteredInto(pk.scratch1, wireValues, pk.g1BIndices, len(pk.InfinityB))
	pk.scratch2 = packFrVectorRegularLEFilteredOutInto(pk.scratch2, wireValues[publicVariables:], publicVariables, common.CommitmentWireIndexesToRemove(commitmentInfo))
	batchMSM, err := bridge.Bridge.MSMBatch(pk.handle, pk.scratch0, pk.scratch1, pk.scratch2)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: batched MSMs: %w", err)
	}
	arBaseAff, err := decodeG1AffineFromPacked(batchMSM.G1ABytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.A: %w", err)
	}
	bs1BaseAff, err := decodeG1AffineFromPacked(batchMSM.G1BBytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.B: %w", err)
	}
	kBaseAff, err := decodeG1AffineFromPacked(batchMSM.G1KBytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.K: %w", err)
	}
	zBaseAff, err := decodeG1AffineFromPacked(zPacked, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G1.Z: %w", err)
	}
	bsBaseAff, err := decodeG2AffineFromPacked(batchMSM.G2BBytes, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu groth16 bn254: msm G2.B: %w", err)
	}

	var r, s big.Int
	var _r, _s, _kr fr.Element
	if _, err := _r.SetRandom(); err != nil {
		return nil, err
	}
	if _, err := _s.SetRandom(); err != nil {
		return nil, err
	}
	_kr.Mul(&_r, &_s).Neg(&_kr)
	_r.BigInt(&r)
	_s.BigInt(&s)

	deltas := bn254.BatchScalarMultiplicationG1(&pk.G1.Delta, []fr.Element{_r, _s, _kr})

	var ar, bs1, krs, krs2, tmp bn254.G1Jac
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

	var bs, deltaS bn254.G2Jac
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

func (pk *ProvingKey) Prepare() error {
	pk.prepareMu.Lock()
	defer pk.prepareMu.Unlock()

	if pk.handle != "" && pk.quotientWarmed {
		return nil
	}
	if err := bridge.Bridge.Init("bn254"); err != nil {
		return err
	}

	if pk.handle == "" {
		payload := bridge.JSObject()
		payload.Set("g1A", bridge.JSUint8Array(packG1AffineJacobianBatch(pk.G1.A)))
		payload.Set("g1ACount", len(pk.G1.A))
		payload.Set("g1B", bridge.JSUint8Array(packG1AffineJacobianBatch(pk.G1.B)))
		payload.Set("g1BCount", len(pk.G1.B))
		payload.Set("g1K", bridge.JSUint8Array(packG1AffineJacobianBatch(pk.G1.K)))
		payload.Set("g1KCount", len(pk.G1.K))
		payload.Set("g1Z", bridge.JSUint8Array(packG1AffineJacobianBatch(pk.G1.Z)))
		payload.Set("g1ZCount", len(pk.G1.Z))
		payload.Set("g2B", bridge.JSUint8Array(packG2AffineJacobianBatch(pk.G2.B)))
		payload.Set("g2BCount", len(pk.G2.B))
		payload.Set("commitmentCount", len(pk.CommitmentKeys))
		for i := range pk.CommitmentKeys {
			suffix := strconv.Itoa(i)
			payload.Set("commitmentBasis"+suffix, bridge.JSUint8Array(packG1AffineJacobianBatch(pk.CommitmentKeys[i].Basis)))
			payload.Set("commitmentBasis"+suffix+"Count", len(pk.CommitmentKeys[i].Basis))
			payload.Set("commitmentBasisExpSigma"+suffix, bridge.JSUint8Array(packG1AffineJacobianBatch(pk.CommitmentKeys[i].BasisExpSigma)))
			payload.Set("commitmentBasisExpSigma"+suffix+"Count", len(pk.CommitmentKeys[i].BasisExpSigma))
		}

		handle, err := bridge.Bridge.PrepareKey("bn254", payload)
		if err != nil {
			return err
		}
		pk.handle = handle
		pk.g1AIndices = common.ComputeKeptIndices(pk.InfinityA)
		pk.g1BIndices = common.ComputeKeptIndices(pk.InfinityB)
	}
	if !pk.quotientWarmed {
		if err := bridge.Bridge.PrewarmQuotientDomain("bn254", int(pk.Domain.Cardinality)); err != nil {
			return err
		}
		pk.quotientWarmed = true
	}
	return nil
}

func packFrVectorRegularLEInto(dst []byte, values []fr.Element) []byte {
	required := len(values) * frBytes
	if cap(dst) < required {
		dst = make([]byte, required)
	} else {
		dst = dst[:required]
	}
	for i := range values {
		base := i * frBytes
		writeFrRegularLE(dst[base:base+frBytes], &values[i])
	}
	return dst
}

func packFrVectorRegularLEFilteredOutInto(dst []byte, values []fr.Element, firstIndex int, remove []int) []byte {
	if len(remove) == 0 {
		return packFrVectorRegularLEInto(dst, values)
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
	required := count * frBytes
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
		writeFrRegularLE(dst[offset:offset+frBytes], &values[i])
		offset += frBytes
	}
	return dst
}

func packFrVectorMontLEPaddedInto(dst []byte, values []fr.Element, size int) []byte {
	required := size * frBytes
	if cap(dst) < required {
		dst = make([]byte, required)
	} else {
		dst = dst[:required]
		clear(dst)
	}
	for i := range values {
		base := i * frBytes
		writeFrMontLE(dst[base:base+frBytes], &values[i])
	}
	return dst
}

func packFrVectorFilteredInto(dst []byte, values []fr.Element, keptPrefixIndices []int, prefixLen int) ([]byte, int) {
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
	required := count * frBytes
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
		writeFrRegularLE(dst[offset:offset+frBytes], &values[idx])
		offset += frBytes
	}
	for i := limit; i < len(values); i++ {
		writeFrRegularLE(dst[offset:offset+frBytes], &values[i])
		offset += frBytes
	}
	return dst, count
}

func writeFrRegularLE(dst []byte, value *fr.Element) {
	be := value.Bytes()
	for i := 0; i < frBytes; i++ {
		dst[i] = be[frBytes-1-i]
	}
}

func writeFrMontLE(dst []byte, value *fr.Element) {
	for i, word := range [4]uint64(*value) {
		binary.LittleEndian.PutUint64(dst[i*8:(i+1)*8], word)
	}
}

func packG1AffineJacobianBatch(points []bn254.G1Affine) []byte {
	out := make([]byte, len(points)*g1PointBytes)
	one := fpOneMontLE()
	for i := range points {
		if points[i].IsInfinity() {
			continue
		}
		base := i * g1PointBytes
		writeFPMontLE(out[base:base+g1CoordinateBytes], &points[i].X)
		writeFPMontLE(out[base+g1CoordinateBytes:base+2*g1CoordinateBytes], &points[i].Y)
		copy(out[base+2*g1CoordinateBytes:base+3*g1CoordinateBytes], one)
	}
	return out
}

func packG2AffineJacobianBatch(points []bn254.G2Affine) []byte {
	out := make([]byte, len(points)*g2PointBytes)
	one := fpOneMontLE()
	for i := range points {
		if points[i].IsInfinity() {
			continue
		}
		base := i * g2PointBytes
		writeFPMontLE(out[base:base+g2ComponentBytes], &points[i].X.A0)
		writeFPMontLE(out[base+g2ComponentBytes:base+2*g2ComponentBytes], &points[i].X.A1)
		writeFPMontLE(out[base+2*g2ComponentBytes:base+3*g2ComponentBytes], &points[i].Y.A0)
		writeFPMontLE(out[base+3*g2ComponentBytes:base+4*g2ComponentBytes], &points[i].Y.A1)
		copy(out[base+4*g2ComponentBytes:base+5*g2ComponentBytes], one)
	}
	return out
}

func decodeG1AffineFromPacked(packed []byte, err error) (bn254.G1Affine, error) {
	if err != nil {
		return bn254.G1Affine{}, err
	}
	if len(packed) != 2*g1CoordinateBytes {
		return bn254.G1Affine{}, fmt.Errorf("webgpu groth16 bn254: expected %d G1 bytes, got %d", 2*g1CoordinateBytes, len(packed))
	}
	return bn254.G1Affine{
		X: readFPMontLE(packed[:g1CoordinateBytes]),
		Y: readFPMontLE(packed[g1CoordinateBytes:]),
	}, nil
}

func decodeG2AffineFromPacked(packed []byte, err error) (bn254.G2Affine, error) {
	if err != nil {
		return bn254.G2Affine{}, err
	}
	if len(packed) != 4*g2ComponentBytes {
		return bn254.G2Affine{}, fmt.Errorf("webgpu groth16 bn254: expected %d G2 bytes, got %d", 4*g2ComponentBytes, len(packed))
	}
	var out bn254.G2Affine
	out.X.A0 = readFPMontLE(packed[0*g2ComponentBytes : 1*g2ComponentBytes])
	out.X.A1 = readFPMontLE(packed[1*g2ComponentBytes : 2*g2ComponentBytes])
	out.Y.A0 = readFPMontLE(packed[2*g2ComponentBytes : 3*g2ComponentBytes])
	out.Y.A1 = readFPMontLE(packed[3*g2ComponentBytes : 4*g2ComponentBytes])
	return out, nil
}

func readFPMontLE(src []byte) fp.Element {
	var words [4]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(src[i*8 : (i+1)*8])
	}
	return fp.Element(words)
}

func writeFPMontLE(dst []byte, value *fp.Element) {
	for i, word := range [4]uint64(*value) {
		binary.LittleEndian.PutUint64(dst[i*8:(i+1)*8], word)
	}
}

func fpOneMontLE() []byte {
	out := make([]byte, g1CoordinateBytes)
	var one fp.Element
	one.SetOne()
	writeFPMontLE(out, &one)
	return out
}
