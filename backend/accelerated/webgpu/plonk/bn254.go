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
	handle               string
	staticNumeratorCache *staticNumeratorCache
}

type staticNumeratorCache struct {
	domain0Cardinality uint64
	domain1Cardinality uint64
	qcpCount           int
	canonical          staticNumeratorPolys
	cosets             []staticNumeratorPolys
}

type staticNumeratorPolys struct {
	ql, qr, qm, qo *iop.Polynomial
	s1, s2, s3     *iop.Polynomial
	qcp            []*iop.Polynomial
}

func proveBN254(spr *cs.SparseR1CS, pk *BN254ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (proof *native.Proof, err error) {
	// parse the options
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("get prover options: %w", err)
	}

	if err := pk.ensurePrepared(); err != nil {
		return nil, fmt.Errorf("prepare proving key: %w", err)
	}

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

	return instance.proof, nil
}

func (pk *BN254ProvingKey) ensureStaticNumeratorCache(trace *native.Trace, domain0, domain1 *fft.Domain) error {
	qcpCount := len(trace.Qcp)
	if pk.staticNumeratorCache != nil &&
		pk.staticNumeratorCache.domain0Cardinality == domain0.Cardinality &&
		pk.staticNumeratorCache.domain1Cardinality == domain1.Cardinality &&
		pk.staticNumeratorCache.qcpCount == qcpCount {
		pk.staticNumeratorCache.canonical.applyToTrace(trace)
		return nil
	}

	canonical := cloneStaticNumeratorPolys(trace)
	canonical.forEach(func(p *iop.Polynomial) {
		p.ToCanonical(domain0).ToRegular()
	})

	rho := int(domain1.Cardinality / domain0.Cardinality)
	cosets := make([]staticNumeratorPolys, rho)

	cosetTable, err := domain0.CosetTable()
	if err != nil {
		return err
	}
	scalingVector := cosetTable
	scalingVectorRev := make([]fr.Element, len(cosetTable))
	copy(scalingVectorRev, cosetTable)
	fft.BitReverse(scalingVectorRev) //nolint:staticcheck // method is backwards compatible

	working := canonical.clone()
	for i := 0; i < rho; i++ {
		if i == 1 {
			w := domain1.Generator
			scalingVector = make([]fr.Element, domain0.Cardinality)
			fft.BuildExpTable(w, scalingVector)

			copy(scalingVectorRev, scalingVector)
			fft.BitReverse(scalingVectorRev) //nolint:staticcheck // method is backwards compatible
		}

		working.forEach(func(p *iop.Polynomial) {
			transformPolynomialToCoset(p, domain0, scalingVector, scalingVectorRev)
		})
		cosets[i] = working.clone()
	}

	pk.staticNumeratorCache = &staticNumeratorCache{
		domain0Cardinality: domain0.Cardinality,
		domain1Cardinality: domain1.Cardinality,
		qcpCount:           qcpCount,
		canonical:          canonical,
		cosets:             cosets,
	}
	pk.staticNumeratorCache.canonical.applyToTrace(trace)
	return nil
}

func cloneStaticNumeratorPolys(trace *native.Trace) staticNumeratorPolys {
	res := staticNumeratorPolys{
		ql:  trace.Ql.Clone(),
		qr:  trace.Qr.Clone(),
		qm:  trace.Qm.Clone(),
		qo:  trace.Qo.Clone(),
		s1:  trace.S1.Clone(),
		s2:  trace.S2.Clone(),
		s3:  trace.S3.Clone(),
		qcp: make([]*iop.Polynomial, len(trace.Qcp)),
	}
	for i := range trace.Qcp {
		res.qcp[i] = trace.Qcp[i].Clone()
	}
	return res
}

func (p staticNumeratorPolys) clone() staticNumeratorPolys {
	res := staticNumeratorPolys{
		ql:  p.ql.Clone(),
		qr:  p.qr.Clone(),
		qm:  p.qm.Clone(),
		qo:  p.qo.Clone(),
		s1:  p.s1.Clone(),
		s2:  p.s2.Clone(),
		s3:  p.s3.Clone(),
		qcp: make([]*iop.Polynomial, len(p.qcp)),
	}
	for i := range p.qcp {
		res.qcp[i] = p.qcp[i].Clone()
	}
	return res
}

func (p staticNumeratorPolys) forEach(fn func(*iop.Polynomial)) {
	fn(p.ql)
	fn(p.qr)
	fn(p.qm)
	fn(p.qo)
	fn(p.s1)
	fn(p.s2)
	fn(p.s3)
	for i := range p.qcp {
		fn(p.qcp[i])
	}
}

func (p staticNumeratorPolys) applyToTrace(trace *native.Trace) {
	trace.Ql = p.ql
	trace.Qr = p.qr
	trace.Qm = p.qm
	trace.Qo = p.qo
	trace.S1 = p.s1
	trace.S2 = p.s2
	trace.S3 = p.s3
	trace.Qcp = p.qcp
}

func (p staticNumeratorPolys) applyToEval(dst []*iop.Polynomial) {
	dst[id_Ql] = p.ql
	dst[id_Qr] = p.qr
	dst[id_Qm] = p.qm
	dst[id_Qo] = p.qo
	dst[id_S1] = p.s1
	dst[id_S2] = p.s2
	dst[id_S3] = p.s3
	for i := range p.qcp {
		dst[id_Qci+2*i] = p.qcp[i]
	}
}

func transformPolynomialToCoset(p *iop.Polynomial, domain *fft.Domain, scalingVector, scalingVectorRev []fr.Element) {
	// shift polynomials to be in the correct coset
	p.ToCanonical(domain)

	// scale by shifter
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
	p.ToLagrange(domain).ToRegular()
}

func (pk *BN254ProvingKey) ensurePrepared() error {
	if pk.handle != "" {
		return nil
	}
	if err := bridgeInit("bn254"); err != nil {
		return err
	}
	payload := jsObject()
	payload.Set("kzg", jsUint8Array(packBN254G1AffineJacobianBatch(pk.Kzg.G1)))
	payload.Set("kzgCount", len(pk.Kzg.G1))
	payload.Set("kzgLagrange", jsUint8Array(packBN254G1AffineJacobianBatch(pk.KzgLagrange.G1)))
	payload.Set("kzgLagrangeCount", len(pk.KzgLagrange.G1))
	handle, err := bridgePrepareKey("bn254", payload)
	if err != nil {
		return err
	}
	pk.handle = handle
	return nil
}

func (pk *BN254ProvingKey) prepareWithCS(spr *cs.SparseR1CS) error {
	if err := pk.ensurePrepared(); err != nil {
		return err
	}
	domain0, domain1 := domainsForSPR(spr)
	trace := native.NewTrace(spr, domain0)
	if err := pk.ensureStaticNumeratorCache(trace, domain0, domain1); err != nil {
		return err
	}
	return bridgePrewarmQuotientTransformDomain("bn254", int(domain0.Cardinality))
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

func readBN254FrRegularLE(src []byte) (fr.Element, error) {
	if len(src) != bn254FrBytes {
		return fr.Element{}, fmt.Errorf("webgpu plonk bn254: expected %d Fr bytes, got %d", bn254FrBytes, len(src))
	}
	var le [bn254FrBytes]byte
	copy(le[:], src)
	return fr.LittleEndian.Element(&le)
}

func unpackBN254FrVectorRegularLEInto(dst []fr.Element, src []byte) error {
	if len(src) != len(dst)*bn254FrBytes {
		return fmt.Errorf("webgpu plonk bn254: expected %d Fr vector bytes, got %d", len(dst)*bn254FrBytes, len(src))
	}
	for i := range dst {
		value, err := readBN254FrRegularLE(src[i*bn254FrBytes : (i+1)*bn254FrBytes])
		if err != nil {
			return err
		}
		dst[i] = value
	}
	return nil
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
	s.domain0, s.domain1 = domainsForSPR(spr)

	// sampling random numbers for blinding the quotient
	if opts.StatisticalZK {
		s.quotientShardsRandomizers[0].SetRandom()
		s.quotientShardsRandomizers[1].SetRandom()
	}

	// build trace
	s.trace = native.NewTrace(spr, s.domain0)
	if err := pk.ensureStaticNumeratorCache(s.trace, s.domain0, s.domain1); err != nil {
		return nil, err
	}

	return &s, nil
}

func domainsForSPR(spr *cs.SparseR1CS) (*fft.Domain, *fft.Domain) {
	nbConstraints := spr.GetNbConstraints()
	sizeSystem := uint64(nbConstraints + len(spr.Public)) // len(spr.Public) is for the placeholder constraints
	domain0 := fft.NewDomain(sizeSystem)

	// h, the quotient polynomial is of degree 3(n+1)+2, so it's in a 3(n+2) dim vector space,
	// the domain is the next power of 2 superior to 3(n+2). 4*domainNum is enough in all cases
	// except when n<6.
	var domain1 *fft.Domain
	if sizeSystem < 6 {
		domain1 = fft.NewDomain(8*sizeSystem, fft.WithoutPrecompute())
	} else {
		domain1 = fft.NewDomain(4*sizeSystem, fft.WithoutPrecompute())
	}
	return domain0, domain1
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

func (s *instance) transformGroupToCoset(ids []int, scalingVector []fr.Element) error {
	polys := make([]*iop.Polynomial, 0, len(ids))
	n := int(s.domain0.Cardinality)
	for _, id := range ids {
		if id >= len(s.x) || id == id_ZS || s.x[id] == nil {
			continue
		}
		if len(s.x[id].Coefficients()) != n {
			return fmt.Errorf("webgpu plonk bn254: quotient polynomial %d has %d coefficients, expected %d", id, len(s.x[id].Coefficients()), n)
		}
		polys = append(polys, s.x[id])
	}
	if len(polys) == 0 {
		return nil
	}

	vectorBytes := n * bn254FrBytes
	valuesPacked := make([]byte, len(polys)*vectorBytes)
	for i, p := range polys {
		packBN254FrVectorRegularLEInto(valuesPacked[i*vectorBytes:(i+1)*vectorBytes], p.Coefficients())
	}
	scalingPacked := packBN254FrVectorRegularLEInto(nil, scalingVector)
	transformedPacked, err := bridgeTransformQuotientCoset("bn254", valuesPacked, scalingPacked, len(polys), n)
	if err != nil {
		return err
	}
	if len(transformedPacked) != len(valuesPacked) {
		return fmt.Errorf("webgpu plonk bn254: quotient transform returned %d bytes, expected %d", len(transformedPacked), len(valuesPacked))
	}
	for i, p := range polys {
		if err := unpackBN254FrVectorRegularLEInto(p.Coefficients(), transformedPacked[i*vectorBytes:(i+1)*vectorBytes]); err != nil {
			return err
		}
		p.Basis = iop.Lagrange
		p.Layout = iop.Regular
	}
	return nil
}

func (s *instance) evaluateQuotientCoset(
	evalX []*iop.Polynomial,
	twiddles0 []fr.Element,
	coset, cosetExpMinusOne, cs, css fr.Element,
	buf []fr.Element,
) error {
	n := int(s.domain0.Cardinality)
	coeffs := func(id int) ([]fr.Element, error) {
		if id >= len(evalX) || evalX[id] == nil {
			return nil, fmt.Errorf("webgpu plonk bn254: missing quotient eval vector %d", id)
		}
		c := evalX[id].Coefficients()
		if len(c) != n {
			return nil, fmt.Errorf("webgpu plonk bn254: quotient eval vector %d has %d elements, expected %d", id, len(c), n)
		}
		return c, nil
	}

	lCoeffs, err := coeffs(id_L)
	if err != nil {
		return err
	}
	rCoeffs, err := coeffs(id_R)
	if err != nil {
		return err
	}
	oCoeffs, err := coeffs(id_O)
	if err != nil {
		return err
	}
	zCoeffs, err := coeffs(id_Z)
	if err != nil {
		return err
	}
	qlCoeffs, err := coeffs(id_Ql)
	if err != nil {
		return err
	}
	qrCoeffs, err := coeffs(id_Qr)
	if err != nil {
		return err
	}
	qmCoeffs, err := coeffs(id_Qm)
	if err != nil {
		return err
	}
	qoCoeffs, err := coeffs(id_Qo)
	if err != nil {
		return err
	}
	qkCoeffs, err := coeffs(id_Qk)
	if err != nil {
		return err
	}
	s1Coeffs, err := coeffs(id_S1)
	if err != nil {
		return err
	}
	s2Coeffs, err := coeffs(id_S2)
	if err != nil {
		return err
	}
	s3Coeffs, err := coeffs(id_S3)
	if err != nil {
		return err
	}

	nbBsbGates := len(s.proof.Bsb22Commitments)
	qcpCoeffs := make([][]fr.Element, nbBsbGates)
	cCommitmentCoeffs := make([][]fr.Element, nbBsbGates)
	for i := 0; i < nbBsbGates; i++ {
		qcpCoeffs[i], err = coeffs(id_Qci + 2*i)
		if err != nil {
			return err
		}
		cCommitmentCoeffs[i], err = coeffs(id_Qci + 2*i + 1)
		if err != nil {
			return err
		}
	}

	blCoeffs := s.bp[id_Bl].Coefficients()
	brCoeffs := s.bp[id_Br].Coefficients()
	boCoeffs := s.bp[id_Bo].Coefficients()
	bzCoeffs := s.bp[id_Bz].Coefficients()

	var one, lagrangeScale fr.Element
	one.SetOne()
	lagrangeScale.Mul(&cosetExpMinusOne, &s.domain0.CardinalityInv)

	for i := 0; i < n; i++ {
		twiddle := &twiddles0[i]
		nextTwiddle := &twiddles0[(i+1)%n]

		l := lCoeffs[i]
		r := rCoeffs[i]
		o := oCoeffs[i]
		z := zCoeffs[i]
		zs := zCoeffs[(i+1)%n]

		var blind fr.Element
		blind = evalSmallPolynomial(blCoeffs, twiddle)
		l.Add(&l, &blind)
		blind = evalSmallPolynomial(brCoeffs, twiddle)
		r.Add(&r, &blind)
		blind = evalSmallPolynomial(boCoeffs, twiddle)
		o.Add(&o, &blind)
		blind = evalSmallPolynomial(bzCoeffs, twiddle)
		z.Add(&z, &blind)
		blind = evalSmallPolynomial(bzCoeffs, nextTwiddle)
		zs.Add(&zs, &blind)

		var gate, tmp fr.Element
		gate.Mul(&qlCoeffs[i], &l)
		tmp.Mul(&qrCoeffs[i], &r)
		gate.Add(&gate, &tmp)
		tmp.Mul(&qmCoeffs[i], &l).Mul(&tmp, &r)
		gate.Add(&gate, &tmp)
		tmp.Mul(&qoCoeffs[i], &o)
		gate.Add(&gate, &tmp).Add(&gate, &qkCoeffs[i])
		for j := 0; j < nbBsbGates; j++ {
			tmp.Mul(&qcpCoeffs[j][i], &cCommitmentCoeffs[j][i])
			gate.Add(&gate, &tmp)
		}

		var id fr.Element
		id.Mul(twiddle, &coset).Mul(&id, &s.beta)

		var a, b, c, right, left fr.Element
		a.Add(&s.gamma, &l).Add(&a, &id)
		b.Mul(&id, &cs).Add(&b, &r).Add(&b, &s.gamma)
		c.Mul(&id, &css).Add(&c, &o).Add(&c, &s.gamma)
		right.Mul(&a, &b).Mul(&right, &c).Mul(&right, &z)

		a.Mul(&s1Coeffs[i], &s.beta).Add(&a, &l).Add(&a, &s.gamma)
		b.Mul(&s2Coeffs[i], &s.beta).Add(&b, &r).Add(&b, &s.gamma)
		c.Mul(&s3Coeffs[i], &s.beta).Add(&c, &o).Add(&c, &s.gamma)
		left.Mul(&a, &b).Mul(&left, &c).Mul(&left, &zs)

		var ordering fr.Element
		ordering.Sub(&left, &right)

		var lone, local fr.Element
		lone.Mul(&lagrangeScale, &s.precomputedDenominators[i])
		local.Sub(&z, &one).Mul(&local, &lone)

		local.Mul(&local, &s.alpha).Add(&local, &ordering).Mul(&local, &s.alpha).Add(&local, &gate)
		buf[i] = local
	}

	return nil
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

	commit, err = s.msmG1("kzgLagrange", 0, p.Coefficients())

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
	if err := s.commitToQuotient(s.h1(), s.h2(), s.h3()); err != nil {
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
	s.proof.ZShiftedOpening, err = s.openKZG(s.blindedZ, zetaShifted)
	if err != nil {
		return err
	}
	return nil
}

func (s *instance) openKZG(p []fr.Element, point fr.Element) (kzg.OpeningProof, error) {
	if len(p) > len(s.pk.Kzg.G1) {
		return kzg.OpeningProof{}, kzg.ErrInvalidPolynomialSize
	}

	var proof kzg.OpeningProof
	proof.ClaimedValue = evalKZGPolynomial(p, point)

	cp := make([]fr.Element, len(p))
	copy(cp, p)
	h := dividePolyByXMinusA(cp, proof.ClaimedValue, point)

	hCommit, err := s.msmG1("kzg", 0, h)
	if err != nil {
		return kzg.OpeningProof{}, err
	}
	proof.H.Set(&hCommit)

	return proof, nil
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
	s.linearizedPolynomialDigest, err = s.msmG1("kzg", 0, s.linearizedPolynomial)
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
	s.proof.BatchedProof, err = s.batchOpenSinglePoint(
		polysToOpen,
		digestsToOpen,
		s.zeta,
		s.kzgFoldingHash,
		s.proof.ZShiftedOpening.ClaimedValue.Marshal(),
	)
	return err
}

func (s *instance) batchOpenSinglePoint(polynomials [][]fr.Element, digests []curve.G1Affine, point fr.Element, hf hash.Hash, dataTranscript ...[]byte) (kzg.BatchOpeningProof, error) {
	nbDigests := len(digests)
	if nbDigests != len(polynomials) {
		return kzg.BatchOpeningProof{}, kzg.ErrInvalidNbDigests
	}
	if nbDigests == 0 {
		return kzg.BatchOpeningProof{}, kzg.ErrZeroNbDigests
	}

	largestPoly := -1
	for _, p := range polynomials {
		if len(p) > len(s.pk.Kzg.G1) {
			return kzg.BatchOpeningProof{}, kzg.ErrInvalidPolynomialSize
		}
		if len(p) > largestPoly {
			largestPoly = len(p)
		}
	}

	var res kzg.BatchOpeningProof
	res.ClaimedValues = make([]fr.Element, len(polynomials))
	for i := range polynomials {
		res.ClaimedValues[i] = evalKZGPolynomial(polynomials[i], point)
	}

	gamma, err := deriveKZGBatchGamma(point, digests, res.ClaimedValues, hf, dataTranscript...)
	if err != nil {
		return kzg.BatchOpeningProof{}, err
	}

	var foldedEvaluations fr.Element
	foldedEvaluations = res.ClaimedValues[nbDigests-1]
	for i := nbDigests - 2; i >= 0; i-- {
		foldedEvaluations.Mul(&foldedEvaluations, &gamma).
			Add(&foldedEvaluations, &res.ClaimedValues[i])
	}

	foldedPolynomials := make([]fr.Element, largestPoly)
	copy(foldedPolynomials, polynomials[0])

	gammaPower := gamma
	for i := 1; i < len(polynomials); i++ {
		var term fr.Element
		for j := range polynomials[i] {
			term.Mul(&polynomials[i][j], &gammaPower)
			foldedPolynomials[j].Add(&foldedPolynomials[j], &term)
		}
		gammaPower.Mul(&gammaPower, &gamma)
	}

	h := dividePolyByXMinusA(foldedPolynomials, foldedEvaluations, point)

	hCommit, err := s.msmG1("kzg", 0, h)
	if err != nil {
		return kzg.BatchOpeningProof{}, err
	}
	res.H.Set(&hCommit)

	return res, nil
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

	var cs, css fr.Element

	// stores the current coset shifter
	var coset fr.Element

	// cosetExponentiatedToNMinusOne stores <coset>^n-1
	var cosetExponentiatedToNMinusOne, one fr.Element
	cs.Set(&s.domain1.FrMultiplicativeGen)
	css.Square(&cs)
	coset.SetOne()
	one.SetOne()
	bn := big.NewInt(int64(n))

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

	// for the first iteration, the scalingVector is the coset table
	scalingVector := cosetTable

	// pre-computed to compute the bit reverse index
	// of the result polynomial
	m := uint64(s.domain1.Cardinality)
	mm := uint64(64 - bits.TrailingZeros64(m))

	s.precomputedDenominators = make([]fr.Element, s.domain0.Cardinality)
	bufBatchInvert := make([]fr.Element, s.domain0.Cardinality)

	staticCache := s.pk.staticNumeratorCache
	if staticCache == nil || len(staticCache.cosets) != rho {
		return nil, errors.New("missing static numerator cache")
	}

	dynamicPolyIDs := []int{id_L, id_R, id_O, id_Z, id_Qk}
	commitmentValuePolyIDs := make([]int, 0, len(s.commitmentInfo))
	for i := range s.commitmentInfo {
		commitmentValuePolyIDs = append(commitmentValuePolyIDs, id_Qci+2*i+1)
	}

	evalX := make([]*iop.Polynomial, len(s.x))
	canonicalizeAndScale := func(p *iop.Polynomial, totalShift fr.Element) {
		p.ToCanonical(s.domain0).ToRegular()
		scalePowers(p, totalShift)
	}
	canonicalizeAndScaleGroup := func(ids []int, totalShift fr.Element) {
		for _, id := range ids {
			if id >= len(s.x) || id == id_ZS || s.x[id] == nil {
				continue
			}
			canonicalizeAndScale(s.x[id], totalShift)
		}
	}

	for i := 0; i < rho; i++ {
		if err := s.track(fmt.Sprintf("quotient_num_coset_%d_total", i), func() error {
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
			}

			if err := s.track(fmt.Sprintf("quotient_num_coset_%d_transform_dynamic", i), func() error {
				return s.transformGroupToCoset(dynamicPolyIDs, scalingVector)
			}); err != nil {
				return err
			}
			if len(commitmentValuePolyIDs) > 0 {
				if err := s.track(fmt.Sprintf("quotient_num_coset_%d_transform_commitment_values", i), func() error {
					return s.transformGroupToCoset(commitmentValuePolyIDs, scalingVector)
				}); err != nil {
					return err
				}
			}

			if err := s.track(fmt.Sprintf("quotient_num_coset_%d_evaluate", i), func() error {
				copy(evalX, s.x)
				staticCache.cosets[i].applyToEval(evalX)
				return s.evaluateQuotientCoset(evalX, twiddles0, coset, cosetExponentiatedToNMinusOne, cs, css, buf)
			}); err != nil {
				return err
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

			return nil
		}); err != nil {
			return nil, err
		}
	}

	// scale everything back
	var totalShift fr.Element
	s.x[id_ZS] = nil
	s.x[id_Qk] = nil

		totalShift.Set(&shifters[0])
		for i := 1; i < len(shifters); i++ {
			totalShift.Mul(&totalShift, &shifters[i])
		}
		totalShift.Inverse(&totalShift)

		canonicalizeAndScaleGroup(dynamicPolyIDs, totalShift)
		if len(commitmentValuePolyIDs) > 0 {
			canonicalizeAndScaleGroup(commitmentValuePolyIDs, totalShift)
		}
		return nil
	}); err != nil {
		return nil, err
	}

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

func evalSmallPolynomial(coeffs []fr.Element, point *fr.Element) fr.Element {
	var res fr.Element
	for i := len(coeffs); i > 0; i-- {
		res.Mul(&res, point).Add(&res, &coeffs[i-1])
	}
	return res
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

func evalKZGPolynomial(p []fr.Element, point fr.Element) fr.Element {
	var res fr.Element
	for i := len(p) - 1; i >= 0; i-- {
		res.Mul(&res, &point).Add(&res, &p[i])
	}
	return res
}

// dividePolyByXMinusA computes (f-f(a))/(x-a), reusing f for the result.
func dividePolyByXMinusA(f []fr.Element, fa, a fr.Element) []fr.Element {
	if len(f) == 0 {
		return []fr.Element{}
	}

	f[0].Sub(&f[0], &fa)

	var t fr.Element
	for i := len(f) - 2; i >= 0; i-- {
		t.Mul(&f[i+1], &a)
		f[i].Add(&f[i], &t)
	}

	return f[1:]
}

func deriveKZGBatchGamma(point fr.Element, digests []curve.G1Affine, claimedValues []fr.Element, hf hash.Hash, dataTranscript ...[]byte) (fr.Element, error) {
	fs := fiatshamir.NewTranscript(hf, "gamma")
	if err := fs.Bind("gamma", point.Marshal()); err != nil {
		return fr.Element{}, err
	}
	for i := range digests {
		if err := fs.Bind("gamma", digests[i].Marshal()); err != nil {
			return fr.Element{}, err
		}
	}
	for i := range claimedValues {
		if err := fs.Bind("gamma", claimedValues[i].Marshal()); err != nil {
			return fr.Element{}, err
		}
	}
	for i := range dataTranscript {
		if err := fs.Bind("gamma", dataTranscript[i]); err != nil {
			return fr.Element{}, err
		}
	}

	gammaBytes, err := fs.ComputeChallenge("gamma")
	if err != nil {
		return fr.Element{}, err
	}
	var gamma fr.Element
	gamma.SetBytes(gammaBytes)
	return gamma, nil
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

func (s *instance) commitToQuotient(h1, h2, h3 []fr.Element) error {
	var err error
	if s.proof.H[0], err = s.msmG1("kzg", 0, h1); err != nil {
		return err
	}
	if s.proof.H[1], err = s.msmG1("kzg", 0, h2); err != nil {
		return err
	}
	s.proof.H[2], err = s.msmG1("kzg", 0, h3)
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
