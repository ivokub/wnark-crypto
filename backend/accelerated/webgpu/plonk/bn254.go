//go:build js && wasm

// Copyright 2020-2026 Consensys Software Inc.
// Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

package plonk

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"math/big"
	"math/bits"
	"time"

	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	bn254fp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/iop"
	"github.com/consensys/gnark-crypto/ecc/bn254/kzg"
	fiatshamir "github.com/consensys/gnark-crypto/fiat-shamir"
	"github.com/consensys/gnark/backend"
	native "github.com/consensys/gnark/backend/plonk/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	fcs "github.com/consensys/gnark/frontend/cs"
	"github.com/consensys/gnark/logger"
)

const (
	id_L int = iota
	id_R
	id_O
	id_Z
	id_ZS
	id_Ql
	id_Qr
	id_Qm
	id_Qo
	id_Qk
	id_S1
	id_S2
	id_S3
	id_Qci // [ .. , Qc_i, Pi_i, ...]
)

// blinding factors
const (
	id_Bl int = iota
	id_Br
	id_Bo
	id_Bz
	nb_blinding_polynomials
)

// blinding orders (-1 to deactivate)
const (
	order_blinding_L = 1
	order_blinding_R = 1
	order_blinding_O = 1
	order_blinding_Z = 2
)

const (
	bn254FrBytes           = fr.Bytes
	bn254G1CoordinateBytes = bn254fp.Bytes
	bn254G1PointBytes      = 3 * bn254G1CoordinateBytes
)

type BN254ProvingKey struct {
	native.ProvingKey
	handle string
}

func proveBN254(spr *cs.SparseR1CS, pk *BN254ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (*native.Proof, error) {

	log := logger.Logger().With().
		Str("curve", spr.CurveID().String()).
		Int("nbConstraints", spr.GetNbConstraints()).
		Str("backend", "plonk").Logger()

	// parse the options
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("get prover options: %w", err)
	}

	if err := pk.ensurePrepared(); err != nil {
		return nil, fmt.Errorf("prepare proving key: %w", err)
	}

	start := time.Now()

	// init instance
	instance, err := newInstance(spr, pk, fullWitness, &opt)
	if err != nil {
		return nil, fmt.Errorf("new instance: %w", err)
	}

	if err := instance.initBlindingPolynomials(); err != nil {
		return nil, fmt.Errorf("init blinding polynomials: %w", err)
	}
	if err := instance.solveConstraints(); err != nil {
		return nil, fmt.Errorf("solve constraints: %w", err)
	}
	if err := instance.completeQk(); err != nil {
		return nil, fmt.Errorf("complete qk: %w", err)
	}
	if err := instance.deriveGammaAndBeta(); err != nil {
		return nil, fmt.Errorf("derive gamma and beta: %w", err)
	}
	if err := instance.buildRatioCopyConstraint(); err != nil {
		return nil, fmt.Errorf("build ratio copy constraint: %w", err)
	}
	if err := instance.computeQuotient(); err != nil {
		return nil, fmt.Errorf("compute quotient: %w", err)
	}
	if err := instance.openZ(); err != nil {
		return nil, fmt.Errorf("open z: %w", err)
	}
	if err := instance.computeLinearizedPolynomial(); err != nil {
		return nil, fmt.Errorf("compute linearized polynomial: %w", err)
	}
	if err := instance.batchOpening(); err != nil {
		return nil, fmt.Errorf("batch opening: %w", err)
	}

	log.Debug().Dur("took", time.Since(start)).Msg("prover done")
	return instance.proof, nil
}

func (pk *BN254ProvingKey) ensurePrepared() error {
	if pk.handle != "" {
		return nil
	}
	if err := bridgeInit("bn254"); err != nil {
		return err
	}
	payload := jsObject()
	payload.Set("kzgLagrange", jsUint8Array(packBN254G1AffineJacobianBatch(pk.KzgLagrange.G1)))
	payload.Set("kzgLagrangeCount", len(pk.KzgLagrange.G1))
	handle, err := bridgePrepareKey("bn254", payload)
	if err != nil {
		return err
	}
	pk.handle = handle
	return nil
}

func packBN254FrVectorRegularLEInto(dst []byte, values []fr.Element) []byte {
	required := len(values) * bn254FrBytes
	if cap(dst) < required {
		dst = make([]byte, required)
	} else {
		dst = dst[:required]
	}
	for i := range values {
		base := i * bn254FrBytes
		writeBN254FrRegularLE(dst[base:base+bn254FrBytes], &values[i])
	}
	return dst
}

func writeBN254FrRegularLE(dst []byte, value *fr.Element) {
	be := value.Bytes()
	for i := 0; i < bn254FrBytes; i++ {
		dst[i] = be[bn254FrBytes-1-i]
	}
}

func packBN254G1AffineJacobianBatch(points []curve.G1Affine) []byte {
	out := make([]byte, len(points)*bn254G1PointBytes)
	for i := range points {
		base := i * bn254G1PointBytes
		writeBN254FPMontLE(out[base:base+bn254G1CoordinateBytes], &points[i].X)
		writeBN254FPMontLE(out[base+bn254G1CoordinateBytes:base+2*bn254G1CoordinateBytes], &points[i].Y)
		writeBN254G1JacobianZOne(out[base+2*bn254G1CoordinateBytes : base+3*bn254G1CoordinateBytes])
	}
	return out
}

func decodeBN254G1AffineFromPacked(packed []byte, err error) (curve.G1Affine, error) {
	if err != nil {
		return curve.G1Affine{}, err
	}
	if len(packed) != 2*bn254G1CoordinateBytes {
		return curve.G1Affine{}, fmt.Errorf("webgpu plonk bn254: expected %d G1 bytes, got %d", 2*bn254G1CoordinateBytes, len(packed))
	}
	return curve.G1Affine{
		X: readBN254FPMontLE(packed[:bn254G1CoordinateBytes]),
		Y: readBN254FPMontLE(packed[bn254G1CoordinateBytes:]),
	}, nil
}

func readBN254FPMontLE(src []byte) bn254fp.Element {
	var words [4]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(src[i*8 : (i+1)*8])
	}
	return bn254fp.Element(words)
}

func writeBN254FPMontLE(dst []byte, value *bn254fp.Element) {
	words := [4]uint64(*value)
	for i := range words {
		binary.LittleEndian.PutUint64(dst[i*8:(i+1)*8], words[i])
	}
}

func writeBN254G1JacobianZOne(dst []byte) {
	var one bn254fp.Element
	one.SetOne()
	writeBN254FPMontLE(dst, &one)
}

// represents a Prover instance
type instance struct {
	pk    *BN254ProvingKey
	proof *native.Proof
	spr   *cs.SparseR1CS
	opt   *backend.ProverConfig

	fs             *fiatshamir.Transcript
	kzgFoldingHash hash.Hash // for KZG folding
	htfFunc        hash.Hash // hash to field function

	// polynomials
	x                         []*iop.Polynomial // x stores tracks the polynomial we need
	bp                        []*iop.Polynomial // blinding polynomials
	h                         *iop.Polynomial   // h is the quotient polynomial
	blindedZ                  []fr.Element      // blindedZ is the blinded version of Z
	quotientShardsRandomizers [2]fr.Element     // random elements for blinding the shards of the quotient

	precomputedDenominators    []fr.Element // stores the denominators of the Lagrange polynomials
	linearizedPolynomial       []fr.Element
	linearizedPolynomialDigest kzg.Digest

	fullWitness witness.Witness

	// bsb22 commitment stuff
	commitmentInfo constraint.PlonkCommitments
	commitmentVal  []fr.Element
	cCommitments   []*iop.Polynomial

	// challenges
	gamma, beta, alpha, zeta fr.Element

	domain0, domain1 *fft.Domain

	trace *native.Trace
}

func newInstance(spr *cs.SparseR1CS, pk *BN254ProvingKey, fullWitness witness.Witness, opts *backend.ProverConfig) (*instance, error) {
	if opts.HashToFieldFn == nil {
		opts.HashToFieldFn = hash_to_field.New([]byte("BSB22-Plonk"))
	}
	s := instance{
		pk:             pk,
		proof:          &native.Proof{},
		spr:            spr,
		opt:            opts,
		fullWitness:    fullWitness,
		bp:             make([]*iop.Polynomial, nb_blinding_polynomials),
		fs:             fiatshamir.NewTranscript(opts.ChallengeHash, "gamma", "beta", "alpha", "zeta"),
		kzgFoldingHash: opts.KZGFoldingHash,
		htfFunc:        opts.HashToFieldFn,
	}
	s.initBSB22Commitments()
	s.x = make([]*iop.Polynomial, id_Qci+2*len(s.commitmentInfo))

	// init fft domains
	nbConstraints := spr.GetNbConstraints()
	sizeSystem := uint64(nbConstraints + len(spr.Public)) // len(spr.Public) is for the placeholder constraints
	s.domain0 = fft.NewDomain(sizeSystem)

	// sampling random numbers for blinding the quotient
	if opts.StatisticalZK {
		s.quotientShardsRandomizers[0].SetRandom()
		s.quotientShardsRandomizers[1].SetRandom()
	}

	// h, the quotient polynomial is of degree 3(n+1)+2, so it's in a 3(n+2) dim vector space,
	// the domain is the next power of 2 superior to 3(n+2). 4*domainNum is enough in all cases
	// except when n<6.
	if sizeSystem < 6 {
		s.domain1 = fft.NewDomain(8*sizeSystem, fft.WithoutPrecompute())
	} else {
		s.domain1 = fft.NewDomain(4*sizeSystem, fft.WithoutPrecompute())
	}

	// build trace
	s.trace = native.NewTrace(spr, s.domain0)

	return &s, nil
}

func (s *instance) initBlindingPolynomials() error {
	s.bp[id_Bl] = getRandomPolynomial(order_blinding_L)
	s.bp[id_Br] = getRandomPolynomial(order_blinding_R)
	s.bp[id_Bo] = getRandomPolynomial(order_blinding_O)
	s.bp[id_Bz] = getRandomPolynomial(order_blinding_Z)
	return nil
}

func (s *instance) initBSB22Commitments() {
	s.commitmentInfo = s.spr.CommitmentInfo.(constraint.PlonkCommitments)
	s.commitmentVal = make([]fr.Element, len(s.commitmentInfo)) // TODO @Tabaie get rid of this
	s.cCommitments = make([]*iop.Polynomial, len(s.commitmentInfo))
	s.proof.Bsb22Commitments = make([]kzg.Digest, len(s.commitmentInfo))

	// override the hint for the commitment constraints
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	s.opt.SolverOpts = append(s.opt.SolverOpts, solver.OverrideHint(bsb22ID, s.bsb22Hint))
}

// Computing and verifying Bsb22 multi-commits explained in https://hackmd.io/x8KsadW3RRyX7YTCFJIkHg
func (s *instance) bsb22Hint(_ *big.Int, ins, outs []*big.Int) error {
	var err error
	commDepth := int(ins[0].Int64())
	ins = ins[1:]

	res := &s.commitmentVal[commDepth]

	commitmentInfo := s.spr.CommitmentInfo.(constraint.PlonkCommitments)[commDepth]
	committedValues := make([]fr.Element, s.domain0.Cardinality)
	offset := s.spr.GetNbPublicVariables()
	for i := range ins {
		committedValues[offset+commitmentInfo.Committed[i]].SetBigInt(ins[i])
	}
	if _, err = committedValues[offset+commitmentInfo.CommitmentIndex].SetRandom(); err != nil { // Commitment injection constraint has qcp = 0. Safe to use for blinding.
		return err
	}
	if _, err = committedValues[offset+s.spr.GetNbConstraints()-1].SetRandom(); err != nil { // Last constraint has qcp = 0. Safe to use for blinding
		return err
	}
	s.cCommitments[commDepth] = iop.NewPolynomial(&committedValues, iop.Form{Basis: iop.Lagrange, Layout: iop.Regular})
	if s.proof.Bsb22Commitments[commDepth], err = kzg.Commit(s.cCommitments[commDepth].Coefficients(), s.pk.KzgLagrange, 1); err != nil {
		return err
	}

	s.htfFunc.Write(s.proof.Bsb22Commitments[commDepth].Marshal())
	hashBts := s.htfFunc.Sum(nil)
	s.htfFunc.Reset()
	nbBuf := fr.Bytes
	if s.htfFunc.Size() < fr.Bytes {
		nbBuf = s.htfFunc.Size()
	}
	res.SetBytes(hashBts[:nbBuf]) // TODO @Tabaie use CommitmentIndex for this; create a new variable CommitmentConstraintIndex for other uses
	res.BigInt(outs[0])

	return nil
}

// solveConstraints computes the evaluation of the polynomials L, R, O
// and sets x[id_L], x[id_R], x[id_O] in Lagrange form
func (s *instance) solveConstraints() error {
	_solution, err := s.spr.Solve(s.fullWitness, s.opt.SolverOpts...)
	if err != nil {
		return err
	}
	solution := _solution.(*cs.SparseR1CSSolution)
	evaluationLDomainSmall := []fr.Element(solution.L)
	evaluationRDomainSmall := []fr.Element(solution.R)
	evaluationODomainSmall := []fr.Element(solution.O)
	s.x[id_L] = iop.NewPolynomial(&evaluationLDomainSmall, iop.Form{Basis: iop.Lagrange, Layout: iop.Regular})
	s.x[id_R] = iop.NewPolynomial(&evaluationRDomainSmall, iop.Form{Basis: iop.Lagrange, Layout: iop.Regular})
	s.x[id_O] = iop.NewPolynomial(&evaluationODomainSmall, iop.Form{Basis: iop.Lagrange, Layout: iop.Regular})

	// commit to l, r, o and add blinding factors
	if err := s.commitToLRO(); err != nil {
		return err
	}
	return nil
}

func (s *instance) completeQk() error {
	qk := s.trace.Qk.Clone()
	qkCoeffs := qk.Coefficients()

	wWitness, ok := s.fullWitness.Vector().(fr.Vector)
	if !ok {
		return witness.ErrInvalidWitness
	}

	copy(qkCoeffs, wWitness[:len(s.spr.Public)])

	for i := range s.commitmentInfo {
		qkCoeffs[s.spr.GetNbPublicVariables()+s.commitmentInfo[i].CommitmentIndex] = s.commitmentVal[i]
	}

	s.x[id_Qk] = qk

	return nil
}

// computeLagrangeOneOnCoset computes 1/n (x**n-1)/(x-1) on coset*ωⁱ
func (s *instance) computeLagrangeOneOnCoset(cosetExpMinusOne fr.Element, index int) fr.Element {
	var res fr.Element
	res.Mul(&cosetExpMinusOne, &s.domain0.CardinalityInv).
		Mul(&res, &s.precomputedDenominators[index])
	return res
}

// commitToLRO commits to L, R, O polynomials using reduced-size MSMs.
//
// L, R, O live on a domain of size n = 2^k, but only offset = nbPublic + nbConstraints
// entries carry actual values. The rest are s0 = witness[0] (first public input).
// For R and O, the first nbPublic entries (placeholders) are also s0.
//
// Key identity: Σ_{i=0}^{n-1} KzgLagrange.G1[i] = [Σ L_i(τ)]₁ = [1]₁ = Kzg.G1[0]
//
// So we can rewrite the commitment as:
//
//	[P] = Σ P[i]·G1_lag[i]
//	    = Σ (P[i]-s0)·G1_lag[i] + s0·Σ G1_lag[i]
//	    = MSM((P[i]-s0), G1_lag[i])  + s0·Kzg.G1[0]
//
// The (P[i]-s0) terms are zero in the padding region, so the MSM only needs
// the non-padding entries. For a 2.2M-constraint circuit on a 4M domain,
// this nearly halves each MSM.
func (s *instance) commitToLRO() error {
	n := int(s.domain0.Cardinality)
	nbPublic := len(s.spr.Public)
	offset := nbPublic + s.spr.GetNbConstraints()

	// s0 = witness[0] = first public input
	wWitness, ok := s.fullWitness.Vector().(fr.Vector)
	if !ok {
		return witness.ErrInvalidWitness
	}
	s0 := wWitness[0]

	// correctionPoint = s0 · [1]₁ = s0 · Kzg.G1[0]
	var s0BigInt big.Int
	s0.BigInt(&s0BigInt)
	var correctionPoint curve.G1Affine
	correctionPoint.ScalarMultiplication(&s.pk.Kzg.G1[0], &s0BigInt)

	// L: subtract s0, MSM on [0:offset], add correction + blinding, restore
	coeffs := s.x[id_L].Coefficients()
	for i := 0; i < offset; i++ {
		coeffs[i].Sub(&coeffs[i], &s0)
	}
	var commit curve.G1Affine
	commit, err := s.msmG1("kzgLagrange", 0, coeffs[:offset])
	if err != nil {
		return err
	}
	for i := 0; i < offset; i++ {
		coeffs[i].Add(&coeffs[i], &s0)
	}
	commit.Add(&commit, &correctionPoint)
	cb := commitBlindingFactor(n, s.bp[id_Bl], s.pk.Kzg)
	s.proof.LRO[0].Add(&commit, &cb)

	// R: subtract s0, MSM on [nbPublic:offset], add correction + blinding, restore
	coeffs = s.x[id_R].Coefficients()
	for i := nbPublic; i < offset; i++ {
		coeffs[i].Sub(&coeffs[i], &s0)
	}
	commit, err = s.msmG1("kzgLagrange", nbPublic, coeffs[nbPublic:offset])
	if err != nil {
		return err
	}
	for i := nbPublic; i < offset; i++ {
		coeffs[i].Add(&coeffs[i], &s0)
	}
	commit.Add(&commit, &correctionPoint)
	cb = commitBlindingFactor(n, s.bp[id_Br], s.pk.Kzg)
	s.proof.LRO[1].Add(&commit, &cb)

	// O: same as R
	coeffs = s.x[id_O].Coefficients()
	for i := nbPublic; i < offset; i++ {
		coeffs[i].Sub(&coeffs[i], &s0)
	}
	commit, err = s.msmG1("kzgLagrange", nbPublic, coeffs[nbPublic:offset])
	if err != nil {
		return err
	}
	for i := nbPublic; i < offset; i++ {
		coeffs[i].Add(&coeffs[i], &s0)
	}
	commit.Add(&commit, &correctionPoint)
	cb = commitBlindingFactor(n, s.bp[id_Bo], s.pk.Kzg)
	s.proof.LRO[2].Add(&commit, &cb)

	return nil
}

func (s *instance) msmG1(vectorName string, start int, scalars []fr.Element) (curve.G1Affine, error) {
	scalarsPacked := packBN254FrVectorRegularLEInto(nil, scalars)
	packed, err := bridgeMSMG1Slice(s.pk.handle, vectorName, start, len(scalars), scalarsPacked)
	return decodeBN254G1AffineFromPacked(packed, err)
}

// deriveGammaAndBeta (copy constraint)
func (s *instance) deriveGammaAndBeta() error {
	wWitness, ok := s.fullWitness.Vector().(fr.Vector)
	if !ok {
		return witness.ErrInvalidWitness
	}

	if err := bindPublicData(s.fs, "gamma", s.pk.Vk, wWitness[:len(s.spr.Public)]); err != nil {
		return err
	}

	gamma, err := deriveRandomness(s.fs, "gamma", &s.proof.LRO[0], &s.proof.LRO[1], &s.proof.LRO[2])
	if err != nil {
		return err
	}

	bbeta, err := s.fs.ComputeChallenge("beta")
	if err != nil {
		return err
	}
	s.gamma = gamma
	s.beta.SetBytes(bbeta)

	return nil
}

// commitToPolyAndBlinding computes the KZG commitment of a polynomial p
// in Lagrange form (large degree)
// and add the contribution of a blinding polynomial b (small degree)
// /!\ The polynomial p is supposed to be in Lagrange form.
func (s *instance) commitToPolyAndBlinding(p, b *iop.Polynomial) (commit curve.G1Affine, err error) {

	commit, err = kzg.Commit(p.Coefficients(), s.pk.KzgLagrange, 1)

	// we add in the blinding contribution
	n := int(s.domain0.Cardinality)
	cb := commitBlindingFactor(n, b, s.pk.Kzg)
	commit.Add(&commit, &cb)

	return
}

func (s *instance) deriveAlpha() (err error) {
	alphaDeps := make([]*curve.G1Affine, len(s.proof.Bsb22Commitments)+1)
	for i := range s.proof.Bsb22Commitments {
		alphaDeps[i] = &s.proof.Bsb22Commitments[i]
	}
	alphaDeps[len(alphaDeps)-1] = &s.proof.Z
	s.alpha, err = deriveRandomness(s.fs, "alpha", alphaDeps...)
	return err
}

func (s *instance) deriveZeta() (err error) {
	s.zeta, err = deriveRandomness(s.fs, "zeta", &s.proof.H[0], &s.proof.H[1], &s.proof.H[2])
	return
}

// computeQuotient computes H
func (s *instance) computeQuotient() (err error) {
	s.x[id_Ql] = s.trace.Ql
	s.x[id_Qr] = s.trace.Qr
	s.x[id_Qm] = s.trace.Qm
	s.x[id_Qo] = s.trace.Qo
	s.x[id_S1] = s.trace.S1
	s.x[id_S2] = s.trace.S2
	s.x[id_S3] = s.trace.S3

	for i := 0; i < len(s.commitmentInfo); i++ {
		s.x[id_Qci+2*i] = s.trace.Qcp[i]
	}

	n := s.domain0.Cardinality
	lone := make([]fr.Element, n)
	lone[0].SetOne()

	for i := 0; i < len(s.commitmentInfo); i++ {
		s.x[id_Qci+2*i+1] = s.cCommitments[i]
	}

	// derive alpha
	if err = s.deriveAlpha(); err != nil {
		return err
	}

	// TODO complete waste of memory find another way to do that
	identity := make([]fr.Element, n)
	identity[1].Set(&s.beta)

	s.x[id_ZS] = s.x[id_Z].ShallowClone().Shift(1)

	numerator, err := s.computeNumerator()
	if err != nil {
		return err
	}

	s.h, err = divideByZH(numerator, [2]*fft.Domain{s.domain0, s.domain1})
	if err != nil {
		return err
	}

	// commit to h
	if err := commitToQuotient(s.h1(), s.h2(), s.h3(), s.proof, s.pk.Kzg); err != nil {
		return err
	}

	if err := s.deriveZeta(); err != nil {
		return err
	}

	return nil
}

func (s *instance) buildRatioCopyConstraint() (err error) {
	// TODO @gbotrel having iop.BuildRatioCopyConstraint return something
	// with capacity = len() + 4 would avoid extra alloc / copy during openZ
	s.x[id_Z], err = iop.BuildRatioCopyConstraint(
		[]*iop.Polynomial{
			s.x[id_L],
			s.x[id_R],
			s.x[id_O],
		},
		s.trace.S,
		s.beta,
		s.gamma,
		iop.Form{Basis: iop.Lagrange, Layout: iop.Regular},
		s.domain0,
	)
	if err != nil {
		return err
	}

	// commit to the blinded version of z
	s.proof.Z, err = s.commitToPolyAndBlinding(s.x[id_Z], s.bp[id_Bz])

	return
}

// open Z (blinded) at ωζ
func (s *instance) openZ() (err error) {
	var zetaShifted fr.Element
	zetaShifted.Mul(&s.zeta, &s.pk.Vk.Generator)
	s.blindedZ = getBlindedCoefficients(s.x[id_Z], s.bp[id_Bz])
	// open z at zeta
	s.proof.ZShiftedOpening, err = kzg.Open(s.blindedZ, zetaShifted, s.pk.Kzg)
	if err != nil {
		return err
	}
	return nil
}

func (s *instance) h1() []fr.Element {
	var h1 []fr.Element
	if !s.opt.StatisticalZK {
		h1 = s.h.Coefficients()[:s.domain0.Cardinality+2]
	} else {
		h1 = make([]fr.Element, s.domain0.Cardinality+3)
		copy(h1, s.h.Coefficients()[:s.domain0.Cardinality+2])
		h1[s.domain0.Cardinality+2].Set(&s.quotientShardsRandomizers[0])
	}
	return h1
}

func (s *instance) h2() []fr.Element {
	var h2 []fr.Element
	if !s.opt.StatisticalZK {
		h2 = s.h.Coefficients()[s.domain0.Cardinality+2 : 2*(s.domain0.Cardinality+2)]
	} else {
		h2 = make([]fr.Element, s.domain0.Cardinality+3)
		copy(h2, s.h.Coefficients()[s.domain0.Cardinality+2:2*(s.domain0.Cardinality+2)])
		h2[0].Sub(&h2[0], &s.quotientShardsRandomizers[0])
		h2[s.domain0.Cardinality+2].Set(&s.quotientShardsRandomizers[1])
	}
	return h2
}

func (s *instance) h3() []fr.Element {
	var h3 []fr.Element
	if !s.opt.StatisticalZK {
		h3 = s.h.Coefficients()[2*(s.domain0.Cardinality+2) : 3*(s.domain0.Cardinality+2)]
	} else {
		h3 = make([]fr.Element, s.domain0.Cardinality+2)
		copy(h3, s.h.Coefficients()[2*(s.domain0.Cardinality+2):3*(s.domain0.Cardinality+2)])
		h3[0].Sub(&h3[0], &s.quotientShardsRandomizers[1])
	}
	return h3
}

func (s *instance) computeLinearizedPolynomial() error {
	qcpzeta := make([]fr.Element, len(s.commitmentInfo))
	for i := range s.commitmentInfo {
		qcpzeta[i] = s.trace.Qcp[i].Evaluate(s.zeta)
	}

	blzeta := evaluateBlinded(s.x[id_L], s.bp[id_Bl], s.zeta)
	brzeta := evaluateBlinded(s.x[id_R], s.bp[id_Br], s.zeta)
	bozeta := evaluateBlinded(s.x[id_O], s.bp[id_Bo], s.zeta)
	bzuzeta := s.proof.ZShiftedOpening.ClaimedValue

	s.linearizedPolynomial = s.innerComputeLinearizedPoly(
		blzeta,
		brzeta,
		bozeta,
		s.alpha,
		s.beta,
		s.gamma,
		s.zeta,
		bzuzeta,
		qcpzeta,
		s.blindedZ,
		coefficients(s.cCommitments),
		s.pk,
	)

	var err error
	s.linearizedPolynomialDigest, err = kzg.Commit(s.linearizedPolynomial, s.pk.Kzg, 1)
	return err
}

func (s *instance) batchOpening() error {
	polysQcp := coefficients(s.trace.Qcp)
	polysToOpen := make([][]fr.Element, 6+len(polysQcp))
	copy(polysToOpen[6:], polysQcp)

	polysToOpen[0] = s.linearizedPolynomial
	polysToOpen[1] = getBlindedCoefficients(s.x[id_L], s.bp[id_Bl])
	polysToOpen[2] = getBlindedCoefficients(s.x[id_R], s.bp[id_Br])
	polysToOpen[3] = getBlindedCoefficients(s.x[id_O], s.bp[id_Bo])
	polysToOpen[4] = s.trace.S1.Coefficients()
	polysToOpen[5] = s.trace.S2.Coefficients()

	digestsToOpen := make([]curve.G1Affine, len(s.pk.Vk.Qcp)+6)
	copy(digestsToOpen[6:], s.pk.Vk.Qcp)

	digestsToOpen[0] = s.linearizedPolynomialDigest
	digestsToOpen[1] = s.proof.LRO[0]
	digestsToOpen[2] = s.proof.LRO[1]
	digestsToOpen[3] = s.proof.LRO[2]
	digestsToOpen[4] = s.pk.Vk.S[0]
	digestsToOpen[5] = s.pk.Vk.S[1]

	var err error
	s.proof.BatchedProof, err = kzg.BatchOpenSinglePoint(
		polysToOpen,
		digestsToOpen,
		s.zeta,
		s.kzgFoldingHash,
		s.pk.Kzg,
		s.proof.ZShiftedOpening.ClaimedValue.Marshal(),
	)

	return err
}

// evaluate the full set of constraints, all polynomials in x are back in
// canonical regular form at the end
func (s *instance) computeNumerator() (*iop.Polynomial, error) {
	// init vectors that are used multiple times throughout the computation
	n := s.domain0.Cardinality
	twiddles0 := make([]fr.Element, n)
	if n == 1 {
		// edge case
		twiddles0[0].SetOne()
	} else {
		twiddles, err := s.domain0.Twiddles()
		if err != nil {
			return nil, err
		}
		copy(twiddles0, twiddles[0])
		w := twiddles0[1]
		for i := len(twiddles[0]); i < len(twiddles0); i++ {
			twiddles0[i].Mul(&twiddles0[i-1], &w)
		}
	}

	nbBsbGates := len(s.proof.Bsb22Commitments)

	gateConstraint := func(u ...fr.Element) fr.Element {

		var ic, tmp fr.Element

		ic.Mul(&u[id_Ql], &u[id_L])
		tmp.Mul(&u[id_Qr], &u[id_R])
		ic.Add(&ic, &tmp)
		tmp.Mul(&u[id_Qm], &u[id_L]).Mul(&tmp, &u[id_R])
		ic.Add(&ic, &tmp)
		tmp.Mul(&u[id_Qo], &u[id_O])
		ic.Add(&ic, &tmp).Add(&ic, &u[id_Qk])
		for i := 0; i < nbBsbGates; i++ {
			tmp.Mul(&u[id_Qci+2*i], &u[id_Qci+2*i+1])
			ic.Add(&ic, &tmp)
		}

		return ic
	}

	var cs, css fr.Element
	cs.Set(&s.domain1.FrMultiplicativeGen)
	css.Square(&cs)

	// stores the current coset shifter
	var coset fr.Element
	coset.SetOne()

	// cosetExponentiatedToNMinusOne stores <coset>^n-1
	var cosetExponentiatedToNMinusOne, one fr.Element
	one.SetOne()
	bn := big.NewInt(int64(n))

	orderingConstraint := func(index int, u ...fr.Element) fr.Element {

		gamma := s.gamma

		// ordering constraint
		var a, b, c, r, l, id fr.Element

		// evaluation of ID at coset*ωⁱ where i:=index
		id.Mul(&twiddles0[index], &coset).Mul(&id, &s.beta)

		a.Add(&gamma, &u[id_L]).Add(&a, &id)
		b.Mul(&id, &cs).Add(&b, &u[id_R]).Add(&b, &gamma)
		c.Mul(&id, &css).Add(&c, &u[id_O]).Add(&c, &gamma)
		r.Mul(&a, &b).Mul(&r, &c).Mul(&r, &u[id_Z])

		a.Add(&u[id_S1], &u[id_L]).Add(&a, &gamma)
		b.Add(&u[id_S2], &u[id_R]).Add(&b, &gamma)
		c.Add(&u[id_S3], &u[id_O]).Add(&c, &gamma)
		l.Mul(&a, &b).Mul(&l, &c).Mul(&l, &u[id_ZS])

		l.Sub(&l, &r)

		return l
	}

	localConstraint := func(index int, u ...fr.Element) fr.Element {
		// local constraint
		var res, lone fr.Element
		lone = s.computeLagrangeOneOnCoset(cosetExponentiatedToNMinusOne, index)
		res.SetOne()
		res.Sub(&u[id_Z], &res).Mul(&res, &lone)

		return res
	}

	rho := int(s.domain1.Cardinality / n)
	shifters := make([]fr.Element, rho)
	shifters[0].Set(&s.domain1.FrMultiplicativeGen)
	for i := 1; i < rho; i++ {
		shifters[i].Set(&s.domain1.Generator)
	}

	cosetTable, err := s.domain0.CosetTable()
	if err != nil {
		return nil, err
	}

	// init the result polynomial & buffer
	cres := make([]fr.Element, s.domain1.Cardinality)
	buf := make([]fr.Element, n)

	allConstraints := func(index int, u ...fr.Element) fr.Element {

		// scale S1, S2, S3 by β
		u[id_S1].Mul(&u[id_S1], &s.beta)
		u[id_S2].Mul(&u[id_S2], &s.beta)
		u[id_S3].Mul(&u[id_S3], &s.beta)

		// blind L, R, O, Z, ZS
		var y fr.Element
		y = s.bp[id_Bl].Evaluate(twiddles0[index])
		u[id_L].Add(&u[id_L], &y)
		y = s.bp[id_Br].Evaluate(twiddles0[index])
		u[id_R].Add(&u[id_R], &y)
		y = s.bp[id_Bo].Evaluate(twiddles0[index])
		u[id_O].Add(&u[id_O], &y)
		y = s.bp[id_Bz].Evaluate(twiddles0[index])
		u[id_Z].Add(&u[id_Z], &y)

		// ZS is shifted by 1; need to get correct twiddle
		y = s.bp[id_Bz].Evaluate(twiddles0[(index+1)%int(n)])
		u[id_ZS].Add(&u[id_ZS], &y)

		a := gateConstraint(u...)
		b := orderingConstraint(index, u...)
		c := localConstraint(index, u...)
		c.Mul(&c, &s.alpha).Add(&c, &b).Mul(&c, &s.alpha).Add(&c, &a)
		return c
	}

	// for the first iteration, the scalingVector is the coset table
	scalingVector := cosetTable
	scalingVectorRev := make([]fr.Element, len(cosetTable))
	copy(scalingVectorRev, cosetTable)
	fft.BitReverse(scalingVectorRev) //nolint:staticcheck // method is backwards compatible

	// pre-computed to compute the bit reverse index
	// of the result polynomial
	m := uint64(s.domain1.Cardinality)
	mm := uint64(64 - bits.TrailingZeros64(m))

	s.precomputedDenominators = make([]fr.Element, s.domain0.Cardinality)
	bufBatchInvert := make([]fr.Element, s.domain0.Cardinality)

	for i := 0; i < rho; i++ {

		coset.Mul(&coset, &shifters[i])
		cosetExponentiatedToNMinusOne.Exp(coset, bn).
			Sub(&cosetExponentiatedToNMinusOne, &one)

		for j := 0; j < int(s.domain0.Cardinality); j++ {
			s.precomputedDenominators[j].
				Mul(&coset, &twiddles0[j]).
				Sub(&s.precomputedDenominators[j], &one)
		}
		batchInvert(s.precomputedDenominators, bufBatchInvert)

		// bl <- bl *( (s*ωⁱ)ⁿ-1 )s
		for _, q := range s.bp {
			cq := q.Coefficients()
			acc := cosetExponentiatedToNMinusOne
			for j := 0; j < len(cq); j++ {
				cq[j].Mul(&cq[j], &acc)
				acc.Mul(&acc, &shifters[i])
			}
		}
		if i == 1 {
			// we have to update the scalingVector; instead of scaling by
			// cosets we scale by the twiddles of the large domain.
			w := s.domain1.Generator
			scalingVector = make([]fr.Element, n)
			fft.BuildExpTable(w, scalingVector)

			// reuse memory
			copy(scalingVectorRev, scalingVector)
			fft.BitReverse(scalingVectorRev) //nolint:staticcheck // method is backwards compatible
		}

		// we do **a lot** of FFT here, but on the small domain.
		// note that for all the polynomials in the proving key
		// (Ql, Qr, Qm, Qo, S1, S2, S3, Qcp, Qc) and ID, LOne
		// we could pre-compute these rho*2 FFTs and store them
		// at the cost of a huge memory footprint.
		batchApply(s.x, func(p *iop.Polynomial) {
			// shift polynomials to be in the correct coset
			p.ToCanonical(s.domain0)

			// scale by shifter[i]
			var w []fr.Element
			if p.Layout == iop.Regular {
				w = scalingVector
			} else {
				w = scalingVectorRev
			}

			cp := p.Coefficients()
			for j := range cp {
				cp[j].Mul(&cp[j], &w[j])
			}

			// fft in the correct coset
			p.ToLagrange(s.domain0).ToRegular()
		})

		if _, err := iop.Evaluate(
			allConstraints,
			buf,
			iop.Form{Basis: iop.Lagrange, Layout: iop.Regular},
			s.x...,
		); err != nil {
			return nil, err
		}
		for j := 0; j < int(n); j++ {
			// we build the polynomial in bit reverse order
			cres[bits.Reverse64(uint64(rho*j+i))>>mm] = buf[j]
		}

		cosetExponentiatedToNMinusOne.
			Inverse(&cosetExponentiatedToNMinusOne)
		// bl <- bl *( (s*ωⁱ)ⁿ-1 )**-1
		for _, q := range s.bp {
			cq := q.Coefficients()
			for j := 0; j < len(cq); j++ {
				cq[j].Mul(&cq[j], &cosetExponentiatedToNMinusOne)
			}
		}
	}

	// scale everything back
	s.x[id_ZS] = nil
	s.x[id_Qk] = nil

	var totalShift fr.Element
	totalShift.Set(&shifters[0])
	for i := 1; i < len(shifters); i++ {
		totalShift.Mul(&totalShift, &shifters[i])
	}
	totalShift.Inverse(&totalShift)

	batchApply(s.x, func(p *iop.Polynomial) {
		if p == nil {
			return
		}
		p.ToCanonical(s.domain0).ToRegular()
		scalePowers(p, totalShift)
	})

	for _, q := range s.bp {
		scalePowers(q, totalShift)
	}

	res := iop.NewPolynomial(&cres, iop.Form{Basis: iop.LagrangeCoset, Layout: iop.BitReverse})

	return res, nil

}

// batchInvert modifies in place vec, with vec[i]<-vec[i]^{-1}, using
// the Montgomery batch inversion trick. We don't use gnark-crypto's batchInvert
// because we want to use a buffer preallocated, to avoid wasting memory.
// /!\ it doesn't check that all vec's inputs or non zero, it is ensured by the size
// of the field /!\
func batchInvert(vec, buf []fr.Element) {
	// local function only, vec and buf are of the same size
	copy(buf, vec)
	for i := 1; i < len(vec); i++ {
		vec[i].Mul(&vec[i], &vec[i-1])
	}
	acc := vec[len(vec)-1]
	acc.Inverse(&acc)
	for i := len(vec) - 1; i > 0; i-- {
		vec[i].Mul(&acc, &vec[i-1])
		acc.Mul(&acc, &buf[i])
	}
	vec[0].Set(&acc)
}

// batchApply executes fn on all polynomials in x except x[id_ZS].
func batchApply(x []*iop.Polynomial, fn func(*iop.Polynomial)) {
	for i := 0; i < len(x); i++ {
		if i == id_ZS {
			continue
		}
		fn(x[i])
	}
}

// p <- <p, (1, w, .., wⁿ) >
// p is supposed to be in canonical form
func scalePowers(p *iop.Polynomial, w fr.Element) {
	var acc fr.Element
	acc.SetOne()
	cp := p.Coefficients()
	for i := 0; i < p.Size(); i++ {
		cp[i].Mul(&cp[i], &acc)
		acc.Mul(&acc, &w)
	}
}

func evaluateBlinded(p, bp *iop.Polynomial, zeta fr.Element) fr.Element {
	// Get the size of the polynomial
	n := big.NewInt(int64(p.Size()))

	var pEvaluatedAtZeta fr.Element

	// Evaluate the polynomial and blinded polynomial at zeta
	pEvaluatedAtZeta = p.Evaluate(zeta)
	bpEvaluatedAtZeta := bp.Evaluate(zeta)

	// Multiply the evaluated blinded polynomial by tempElement
	var t fr.Element
	one := fr.One()
	t.Exp(zeta, n).Sub(&t, &one)
	bpEvaluatedAtZeta.Mul(&bpEvaluatedAtZeta, &t)

	// Add the evaluated polynomial and the evaluated blinded polynomial
	pEvaluatedAtZeta.Add(&pEvaluatedAtZeta, &bpEvaluatedAtZeta)

	// Return the result
	return pEvaluatedAtZeta
}

// /!\ modifies the size
func getBlindedCoefficients(p, bp *iop.Polynomial) []fr.Element {
	cp := p.Coefficients()
	cbp := bp.Coefficients()
	cp = append(cp, cbp...)
	for i := 0; i < len(cbp); i++ {
		cp[i].Sub(&cp[i], &cbp[i])
	}
	return cp
}

// commits to a polynomial of the form b*(Xⁿ-1) where b is of small degree
func commitBlindingFactor(n int, b *iop.Polynomial, key kzg.ProvingKey) curve.G1Affine {
	cp := b.Coefficients()
	np := b.Size()

	var res curve.G1Affine
	for i := 0; i < np; i++ {
		var scalar big.Int
		cp[i].BigInt(&scalar)

		var hi, lo curve.G1Affine
		hi.ScalarMultiplication(&key.G1[n+i], &scalar)
		lo.ScalarMultiplication(&key.G1[i], &scalar)
		hi.Sub(&hi, &lo)
		res.Add(&res, &hi)
	}
	return res
}

// return a random polynomial of degree n, if n==-1 cancel the blinding
func getRandomPolynomial(n int) *iop.Polynomial {
	var a []fr.Element
	if n == -1 {
		a = make([]fr.Element, 1)
		a[0].SetZero()
	} else {
		a = make([]fr.Element, n+1)
		for i := 0; i <= n; i++ {
			a[i].SetRandom()
		}
	}
	res := iop.NewPolynomial(&a, iop.Form{
		Basis: iop.Canonical, Layout: iop.Regular})
	return res
}

func coefficients(p []*iop.Polynomial) [][]fr.Element {
	res := make([][]fr.Element, len(p))
	for i, pI := range p {
		res[i] = pI.Coefficients()
	}
	return res
}

func commitToQuotient(h1, h2, h3 []fr.Element, proof *native.Proof, kzgPk kzg.ProvingKey) error {
	var err error
	if proof.H[0], err = kzg.Commit(h1, kzgPk, 1); err != nil {
		return err
	}
	if proof.H[1], err = kzg.Commit(h2, kzgPk, 1); err != nil {
		return err
	}
	proof.H[2], err = kzg.Commit(h3, kzgPk, 1)
	return err
}

// divideByZH
// The input must be in LagrangeCoset.
// The result is in Canonical Regular. (in place using a)
func divideByZH(a *iop.Polynomial, domains [2]*fft.Domain) (*iop.Polynomial, error) {
	smallDomain, bigDomain := domains[0], domains[1]
	if smallDomain == nil || bigDomain == nil {
		return nil, errors.New("invalid domain")
	}
	if smallDomain.Cardinality == 0 || bigDomain.Cardinality == 0 {
		return nil, errors.New("invalid domain cardinality")
	}
	if bigDomain.Cardinality%smallDomain.Cardinality != 0 {
		return nil, errors.New("invalid domain ratio")
	}

	// check that the basis is LagrangeCoset
	if a.Basis != iop.LagrangeCoset || a.Layout != iop.BitReverse {
		return nil, errors.New("invalid form")
	}

	// prepare the evaluations of x^n-1 on the big domain's coset
	xnMinusOneInverseLagrangeCoset := evaluateXnMinusOneDomainBigCoset(domains)
	rho := int(bigDomain.Cardinality / smallDomain.Cardinality)

	r := a.Coefficients()
	n := uint64(len(r))
	nn := uint64(64 - bits.TrailingZeros64(n))

	for i := range r {
		iRev := bits.Reverse64(uint64(i)) >> nn
		r[i].Mul(&r[i], &xnMinusOneInverseLagrangeCoset[int(iRev)%rho])
	}

	// since a is in bit reverse order, ToRegular shouldn't do anything
	a.ToCanonical(bigDomain).ToRegular()

	return a, nil

}

// evaluateXnMinusOneDomainBigCoset evaluates Xᵐ-1 on DomainBig coset
func evaluateXnMinusOneDomainBigCoset(domains [2]*fft.Domain) []fr.Element {

	rho := domains[1].Cardinality / domains[0].Cardinality

	res := make([]fr.Element, rho)

	expo := big.NewInt(int64(domains[0].Cardinality))
	res[0].Exp(domains[1].FrMultiplicativeGen, expo)

	var t fr.Element
	t.Exp(domains[1].Generator, expo)

	one := fr.One()

	for i := 1; i < int(rho); i++ {
		res[i].Mul(&res[i-1], &t)
		res[i-1].Sub(&res[i-1], &one)
	}
	res[len(res)-1].Sub(&res[len(res)-1], &one)

	res = fr.BatchInvert(res)

	return res
}

// innerComputeLinearizedPoly computes the linearized polynomial in canonical basis.
// The purpose is to commit and open all in one ql, qr, qm, qo, qk.
// * lZeta, rZeta, oZeta are the evaluation of l, r, o at zeta
// * z is the permutation polynomial, zu is Z(μX), the shifted version of Z
// * pk is the proving key: the linearized polynomial is a linear combination of ql, qr, qm, qo, qk.
//
// The Linearized polynomial is:
//
// α²*L₁(ζ)*Z(X)
// + α*( (l(ζ)+β*s1(ζ)+γ)*(r(ζ)+β*s2(ζ)+γ)*(β*s3(X))*Z(μζ) - Z(X)*(l(ζ)+β*id1(ζ)+γ)*(r(ζ)+β*id2(ζ)+γ)*(o(ζ)+β*id3(ζ)+γ))
// + l(ζ)*Ql(X) + l(ζ)r(ζ)*Qm(X) + r(ζ)*Qr(X) + o(ζ)*Qo(X) + Qk(X) + ∑ᵢQcp_(ζ)Pi_(X)
// - Z_{H}(ζ)*((H₀(X) + ζᵐ⁺²*H₁(X) + ζ²⁽ᵐ⁺²⁾*H₂(X))
//
// /!\ blindedZCanonical is modified
func (s *instance) innerComputeLinearizedPoly(lZeta, rZeta, oZeta, alpha, beta, gamma, zeta, zu fr.Element, qcpZeta, blindedZCanonical []fr.Element, pi2Canonical [][]fr.Element, pk *BN254ProvingKey) []fr.Element {

	// l(ζ)r(ζ)
	var rl fr.Element
	rl.Mul(&rZeta, &lZeta)

	// s1 =  α*(l(ζ)+β*s1(β)+γ)*(r(ζ)+β*s2(β)+γ)*β*Z(μζ)
	// s2 = -α*(l(ζ)+β*ζ+γ)*(r(ζ)+β*u*ζ+γ)*(o(ζ)+β*u²*ζ+γ)
	// the linearised polynomial is
	// α²*L₁(ζ)*Z(X) +
	// s1*s3(X)+s2*Z(X) + l(ζ)*Ql(X) +
	// l(ζ)r(ζ)*Qm(X) + r(ζ)*Qr(X) + o(ζ)*Qo(X) + Qk(X) + ∑ᵢQcp_(ζ)Pi_(X) -
	// Z_{H}(ζ)*((H₀(X) + ζᵐ⁺²*H₁(X) + ζ²⁽ᵐ⁺²⁾*H₂(X))
	var s1, s2 fr.Element
	s1 = s.trace.S1.Evaluate(zeta)                                   // s1(ζ)
	s1.Mul(&s1, &beta).Add(&s1, &lZeta).Add(&s1, &gamma)             // (l(ζ)+β*s1(ζ)+γ)
	tmp := s.trace.S2.Evaluate(zeta)                                 // s2(ζ)
	tmp.Mul(&tmp, &beta).Add(&tmp, &rZeta).Add(&tmp, &gamma)         // (r(ζ)+β*s2(ζ)+γ)
	s1.Mul(&s1, &tmp).Mul(&s1, &zu).Mul(&s1, &beta).Mul(&s1, &alpha) // (l(ζ)+β*s1(ζ)+γ)*(r(ζ)+β*s2(ζ)+γ)*β*Z(μζ)*α

	var uzeta, uuzeta fr.Element
	uzeta.Mul(&zeta, &pk.Vk.CosetShift)
	uuzeta.Mul(&uzeta, &pk.Vk.CosetShift)

	s2.Mul(&beta, &zeta).Add(&s2, &lZeta).Add(&s2, &gamma)      // (l(ζ)+β*ζ+γ)
	tmp.Mul(&beta, &uzeta).Add(&tmp, &rZeta).Add(&tmp, &gamma)  // (r(ζ)+β*u*ζ+γ)
	s2.Mul(&s2, &tmp)                                           // (l(ζ)+β*ζ+γ)*(r(ζ)+β*u*ζ+γ)
	tmp.Mul(&beta, &uuzeta).Add(&tmp, &oZeta).Add(&tmp, &gamma) // (o(ζ)+β*u²*ζ+γ)
	s2.Mul(&s2, &tmp)                                           // (l(ζ)+β*ζ+γ)*(r(ζ)+β*u*ζ+γ)*(o(ζ)+β*u²*ζ+γ)
	s2.Neg(&s2).Mul(&s2, &alpha)

	// Z_h(ζ), ζⁿ⁺², L₁(ζ)*α²*Z
	var zhZeta, zetaNPlusTwo, alphaSquareLagrangeZero, one, den, frNbElmt fr.Element
	one.SetOne()
	nbElmt := int64(s.domain0.Cardinality)
	alphaSquareLagrangeZero.Set(&zeta).Exp(alphaSquareLagrangeZero, big.NewInt(nbElmt)) // ζⁿ
	zetaNPlusTwo.Mul(&alphaSquareLagrangeZero, &zeta).Mul(&zetaNPlusTwo, &zeta)         // ζⁿ⁺²
	alphaSquareLagrangeZero.Sub(&alphaSquareLagrangeZero, &one)                         // ζⁿ - 1
	zhZeta.Set(&alphaSquareLagrangeZero)                                                // Z_h(ζ) = ζⁿ - 1
	frNbElmt.SetUint64(uint64(nbElmt))
	den.Sub(&zeta, &one).Inverse(&den)                           // 1/(ζ-1)
	alphaSquareLagrangeZero.Mul(&alphaSquareLagrangeZero, &den). // L₁ = (ζⁿ - 1)/(ζ-1)
									Mul(&alphaSquareLagrangeZero, &alpha).
									Mul(&alphaSquareLagrangeZero, &alpha).
									Mul(&alphaSquareLagrangeZero, &s.domain0.CardinalityInv) // α²*L₁(ζ)

	s3canonical := s.trace.S3.Coefficients()

	s.trace.Qk.ToCanonical(s.domain0).ToRegular()

	// len(h1)=len(h2)=len(blindedZCanonical)=len(h3)+1 when Statistical ZK is activated
	// len(h1)=len(h2)=len(h3)=len(blindedZCanonical)-1 when Statistical ZK is deactivated
	h1 := s.h1()
	h2 := s.h2()
	h3 := s.h3()

	// at this stage we have
	// s1 =  α*(l(ζ)+β*s1(β)+γ)*(r(ζ)+β*s2(β)+γ)*β*Z(μζ)
	// s2 = -α*(l(ζ)+β*ζ+γ)*(r(ζ)+β*u*ζ+γ)*(o(ζ)+β*u²*ζ+γ)
	cql := s.trace.Ql.Coefficients()
	cqr := s.trace.Qr.Coefficients()
	cqm := s.trace.Qm.Coefficients()
	cqo := s.trace.Qo.Coefficients()
	cqk := s.trace.Qk.Coefficients()

	var t, t0, t1 fr.Element

	for i := range blindedZCanonical {
		t.Mul(&blindedZCanonical[i], &s2) // -Z(X)*α*(l(ζ)+β*ζ+γ)*(r(ζ)+β*u*ζ+γ)*(o(ζ)+β*u²*ζ+γ)
		if i < len(s3canonical) {
			t0.Mul(&s3canonical[i], &s1) // α*(l(ζ)+β*s1(β)+γ)*(r(ζ)+β*s2(β)+γ)*β*Z(μζ)*β*s3(X)
			t.Add(&t, &t0)
		}
		if i < len(cqm) {
			t1.Mul(&cqm[i], &rl)     // l(ζ)r(ζ)*Qm(X)
			t.Add(&t, &t1)           // linPol += l(ζ)r(ζ)*Qm(X)
			t0.Mul(&cql[i], &lZeta)  // l(ζ)Q_l(X)
			t.Add(&t, &t0)           // linPol += l(ζ)*Ql(X)
			t0.Mul(&cqr[i], &rZeta)  //r(ζ)*Qr(X)
			t.Add(&t, &t0)           // linPol += r(ζ)*Qr(X)
			t0.Mul(&cqo[i], &oZeta)  // o(ζ)*Qo(X)
			t.Add(&t, &t0)           // linPol += o(ζ)*Qo(X)
			t.Add(&t, &cqk[i])       // linPol += Qk(X)
			for j := range qcpZeta { // linPol += ∑ᵢQcp_(ζ)Pi_(X)
				t0.Mul(&pi2Canonical[j][i], &qcpZeta[j])
				t.Add(&t, &t0)
			}
		}

		t0.Mul(&blindedZCanonical[i], &alphaSquareLagrangeZero) // α²L₁(ζ)Z(X)
		blindedZCanonical[i].Add(&t, &t0)                       // linPol += α²L₁(ζ)Z(X)

		// if statistical zeroknowledge is deactivated, len(h1)=len(h2)=len(h3)=len(blindedZ)-1.
		// Else len(h1)=len(h2)=len(blindedZCanonical)=len(h3)+1
		if i < len(h3) {
			t.Mul(&h3[i], &zetaNPlusTwo).
				Add(&t, &h2[i]).
				Mul(&t, &zetaNPlusTwo).
				Add(&t, &h1[i]).
				Mul(&t, &zhZeta)
			blindedZCanonical[i].Sub(&blindedZCanonical[i], &t) // linPol -= Z_h(ζ)*(H₀(X) + ζᵐ⁺²*H₁(X) + ζ²⁽ᵐ⁺²⁾*H₂(X))
		} else if s.opt.StatisticalZK {
			t.Mul(&h2[i], &zetaNPlusTwo).
				Add(&t, &h1[i]).
				Mul(&t, &zhZeta)
			blindedZCanonical[i].Sub(&blindedZCanonical[i], &t) // linPol -= Z_h(ζ)*(H₀(X) + ζᵐ⁺²*H₁(X) + ζ²⁽ᵐ⁺²⁾*H₂(X))
		}
	}

	return blindedZCanonical
}

func bindPublicData(fs *fiatshamir.Transcript, challenge string, vk *native.VerifyingKey, publicInputs []fr.Element) error {
	if err := fs.Bind(challenge, vk.S[0].Marshal()); err != nil {
		return err
	}
	if err := fs.Bind(challenge, vk.S[1].Marshal()); err != nil {
		return err
	}
	if err := fs.Bind(challenge, vk.S[2].Marshal()); err != nil {
		return err
	}

	if err := fs.Bind(challenge, vk.Ql.Marshal()); err != nil {
		return err
	}
	if err := fs.Bind(challenge, vk.Qr.Marshal()); err != nil {
		return err
	}
	if err := fs.Bind(challenge, vk.Qm.Marshal()); err != nil {
		return err
	}
	if err := fs.Bind(challenge, vk.Qo.Marshal()); err != nil {
		return err
	}
	if err := fs.Bind(challenge, vk.Qk.Marshal()); err != nil {
		return err
	}
	for i := range vk.Qcp {
		if err := fs.Bind(challenge, vk.Qcp[i].Marshal()); err != nil {
			return err
		}
	}

	for i := 0; i < len(publicInputs); i++ {
		if err := fs.Bind(challenge, publicInputs[i].Marshal()); err != nil {
			return err
		}
	}

	return nil
}

func deriveRandomness(fs *fiatshamir.Transcript, challenge string, points ...*curve.G1Affine) (fr.Element, error) {
	var buf [curve.SizeOfG1AffineUncompressed]byte
	var r fr.Element

	for _, p := range points {
		buf = p.RawBytes()
		if err := fs.Bind(challenge, buf[:]); err != nil {
			return r, err
		}
	}

	b, err := fs.ComputeChallenge(challenge)
	if err != nil {
		return r, err
	}
	r.SetBytes(b)
	return r, nil
}
