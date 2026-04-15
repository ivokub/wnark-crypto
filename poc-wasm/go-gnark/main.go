//go:build js && wasm

package main

import (
	"encoding/binary"
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	gnarkbls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381"
	gnarkbls12381fp "github.com/consensys/gnark-crypto/ecc/bls12-381/fp"
	gnarkbls12381fr "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	gnarkbls12381fft "github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	gnarkbn254 "github.com/consensys/gnark-crypto/ecc/bn254"
	gnarkbn254fp "github.com/consensys/gnark-crypto/ecc/bn254/fp"
	gnarkbn254fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
	gnarkbn254fft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/utils"
	"github.com/ivokub/wnark-crypto/poc-wasm/internal/pocutil"
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

	nttSize := 1 << cfg.NTTLog
	g1MSMSize := 1 << cfg.G1MSMLog
	g2MSMSize := 1 << cfg.G2MSMLog

	pocutil.Logf("=== Go Wasm -> gnark-crypto (%s) ===", cfg.Curve)
	pocutil.Logf("ntt_size = %d", nttSize)
	pocutil.Logf("ntt_runs = %d", cfg.NTTRuns)
	pocutil.Logf("g1_msm_size = %d", g1MSMSize)
	pocutil.Logf("g1_msm_runs = %d", cfg.G1MSMRuns)
	pocutil.Logf("g2_msm_size = %d", g2MSMSize)
	pocutil.Logf("g2_msm_runs = %d", cfg.G2MSMRuns)

	overallStart := pocutil.NowMS()

	var g1FixtureBytes []byte
	var g2FixtureBytes []byte
	g1FixtureLoadMs := 0.0
	g2FixtureLoadMs := 0.0
	if cfg.G1FixtureBinPath != "" {
		pocutil.SetStatus("Loading G1 fixture")
		start := pocutil.NowMS()
		g1FixtureBytes, err = pocutil.FetchBytes(cfg.G1FixtureBinPath)
		if err != nil {
			return err
		}
		g1FixtureLoadMs = pocutil.NowMS() - start
		pocutil.Logf("g1_fixture_load_ms = %.3f", g1FixtureLoadMs)
	}
	if cfg.G2FixtureBinPath != "" {
		pocutil.SetStatus("Loading G2 fixture")
		start := pocutil.NowMS()
		g2FixtureBytes, err = pocutil.FetchBytes(cfg.G2FixtureBinPath)
		if err != nil {
			return err
		}
		g2FixtureLoadMs = pocutil.NowMS() - start
		pocutil.Logf("g2_fixture_load_ms = %.3f", g2FixtureLoadMs)
	}
	fixtureLoadMs := g1FixtureLoadMs + g2FixtureLoadMs
	pocutil.Logf("startup_ms = %.3f", fixtureLoadMs)

	steadyStateStart := pocutil.NowMS()
	pocutil.SetStatus("Running NTT")

	var nttDigest string
	var nttDuration float64
	switch cfg.Curve {
	case "bn254":
		nttDigest, nttDuration, err = runBN254NTTBatch(nttSize, cfg.NTTRuns)
	case "bls12_381":
		nttDigest, nttDuration, err = runBLS12381NTTBatch(nttSize, cfg.NTTRuns)
	default:
		err = fmt.Errorf("unsupported curve %q", cfg.Curve)
	}
	if err != nil {
		return err
	}
	pocutil.Logf("ntt_total_ms = %.3f", nttDuration)
	pocutil.Logf("ntt_avg_ms = %.3f", nttDuration/float64(cfg.NTTRuns))
	pocutil.Logf("ntt_batch_digest = %s", nttDigest)

	pocutil.SetStatus("Running G1 MSM batch")
	var g1MSMDigest string
	var g1MSMDuration float64
	switch cfg.Curve {
	case "bn254":
		g1MSMDigest, g1MSMDuration, err = runBN254G1MSMBatch(g1MSMSize, cfg.G1MSMRuns, g1FixtureBytes)
	case "bls12_381":
		g1MSMDigest, g1MSMDuration, err = runBLS12381G1MSMBatch(g1MSMSize, cfg.G1MSMRuns, g1FixtureBytes)
	default:
		err = fmt.Errorf("unsupported curve %q", cfg.Curve)
	}
	if err != nil {
		return err
	}
	pocutil.Logf("g1_msm_total_ms = %.3f", g1MSMDuration)
	pocutil.Logf("g1_msm_avg_ms = %.3f", g1MSMDuration/float64(cfg.G1MSMRuns))
	pocutil.Logf("g1_msm_batch_digest = %s", g1MSMDigest)

	pocutil.SetStatus("Running G2 MSM batch")
	var g2MSMDigest string
	var g2MSMDuration float64
	switch cfg.Curve {
	case "bn254":
		g2MSMDigest, g2MSMDuration, err = runBN254G2MSMBatch(g2MSMSize, cfg.G2MSMRuns, g2FixtureBytes)
	case "bls12_381":
		g2MSMDigest, g2MSMDuration, err = runBLS12381G2MSMBatch(g2MSMSize, cfg.G2MSMRuns, g2FixtureBytes)
	default:
		err = fmt.Errorf("unsupported curve %q", cfg.Curve)
	}
	if err != nil {
		return err
	}
	pocutil.Logf("g2_msm_total_ms = %.3f", g2MSMDuration)
	pocutil.Logf("g2_msm_avg_ms = %.3f", g2MSMDuration/float64(cfg.G2MSMRuns))
	pocutil.Logf("g2_msm_batch_digest = %s", g2MSMDigest)

	steadyStateDuration := pocutil.NowMS() - steadyStateStart
	overallDuration := pocutil.NowMS() - overallStart
	pocutil.Logf("steady_state_total_ms = %.3f", steadyStateDuration)
	pocutil.Logf("overall_total_ms = %.3f", overallDuration)

	pocutil.Complete(map[string]any{
		"impl":                     "gnark-go",
		"curve":                    cfg.Curve,
		"fixture_load_ms":          fixtureLoadMs,
		"g1_fixture_load_ms":       g1FixtureLoadMs,
		"g2_fixture_load_ms":       g2FixtureLoadMs,
		"startup_duration_ms":      fixtureLoadMs,
		"steady_state_duration_ms": steadyStateDuration,
		"overall_duration_ms":      overallDuration,
		"ntt_size":                 nttSize,
		"ntt_runs":                 cfg.NTTRuns,
		"ntt_duration_ms":          nttDuration,
		"ntt_digest_hex":           nttDigest,
		"g1_msm_size":              g1MSMSize,
		"g1_msm_runs":              cfg.G1MSMRuns,
		"g1_msm_duration_ms":       g1MSMDuration,
		"g1_msm_digest_hex":        g1MSMDigest,
		"g2_msm_size":              g2MSMSize,
		"g2_msm_runs":              cfg.G2MSMRuns,
		"g2_msm_duration_ms":       g2MSMDuration,
		"g2_msm_digest_hex":        g2MSMDigest,
		"total_duration_ms":        overallDuration,
	})
	return nil
}

func runBN254NTTBatch(size int, runs int) (string, float64, error) {
	domain := gnarkbn254fft.NewDomain(uint64(size))
	digests := make([]byte, 0, runs*64)
	total := 0.0
	for run := 0; run < runs; run++ {
		roundStart := pocutil.NowMS()
		values := make([]gnarkbn254fr.Element, size)
		offset := uint64(run*size + 1)
		for i := range values {
			values[i].SetUint64(offset + uint64(i))
		}
		utils.BitReverse(values)
		domain.FFT(values, gnarkbn254fft.DIT)
		buf := make([]byte, 0, size*gnarkbn254fr.Bytes)
		for i := range values {
			be := values[i].Bytes()
			buf = append(buf, pocutil.ReverseCopy(be[:])...)
		}
		digests = append(digests, []byte(pocutil.SHA256Hex(buf))...)
		roundDuration := pocutil.NowMS() - roundStart
		total += roundDuration
		pocutil.Logf("ntt_round_%d_ms = %.3f", run, roundDuration)
	}
	return pocutil.SHA256Hex(digests), total, nil
}

func runBLS12381NTTBatch(size int, runs int) (string, float64, error) {
	domain := gnarkbls12381fft.NewDomain(uint64(size))
	digests := make([]byte, 0, runs*64)
	total := 0.0
	for run := 0; run < runs; run++ {
		roundStart := pocutil.NowMS()
		values := make([]gnarkbls12381fr.Element, size)
		offset := uint64(run*size + 1)
		for i := range values {
			values[i].SetUint64(offset + uint64(i))
		}
		utils.BitReverse(values)
		domain.FFT(values, gnarkbls12381fft.DIT)
		buf := make([]byte, 0, size*gnarkbls12381fr.Bytes)
		for i := range values {
			be := values[i].Bytes()
			buf = append(buf, pocutil.ReverseCopy(be[:])...)
		}
		digests = append(digests, []byte(pocutil.SHA256Hex(buf))...)
		roundDuration := pocutil.NowMS() - roundStart
		total += roundDuration
		pocutil.Logf("ntt_round_%d_ms = %.3f", run, roundDuration)
	}
	return pocutil.SHA256Hex(digests), total, nil
}

func runBN254G1MSMBatch(size int, runs int, fixture []byte) (string, float64, error) {
	pointBytes := 96
	if len(fixture) < size*pointBytes {
		return "", 0, fmt.Errorf("fixture has %d points, need %d", len(fixture)/pointBytes, size)
	}
	bases := decodeBN254Bases(fixture, size)
	digests := make([]byte, 0, runs*64)
	total := 0.0
	for run := 0; run < runs; run++ {
		roundStart := pocutil.NowMS()
		scalars := make([]gnarkbn254fr.Element, size)
		offset := uint64(run*size + 1)
		for i := range scalars {
			scalars[i].SetUint64(offset + uint64(i))
		}
		result, err := new(gnarkbn254.G1Affine).MultiExp(bases, scalars, ecc.MultiExpConfig{NbTasks: 1})
		if err != nil {
			return "", 0, err
		}
		digestBytes := make([]byte, 0, 64)
		digestBytes = append(digestBytes, bn254FpMontLE(result.X)...)
		digestBytes = append(digestBytes, bn254FpMontLE(result.Y)...)
		digests = append(digests, []byte(pocutil.SHA256Hex(digestBytes))...)
		roundDuration := pocutil.NowMS() - roundStart
		total += roundDuration
		pocutil.Logf("g1_msm_round_%d_ms = %.3f", run, roundDuration)
	}
	return pocutil.SHA256Hex(digests), total, nil
}

func runBLS12381G1MSMBatch(size int, runs int, fixture []byte) (string, float64, error) {
	pointBytes := 144
	if len(fixture) < size*pointBytes {
		return "", 0, fmt.Errorf("fixture has %d points, need %d", len(fixture)/pointBytes, size)
	}
	bases := decodeBLS12381Bases(fixture, size)
	digests := make([]byte, 0, runs*64)
	total := 0.0
	for run := 0; run < runs; run++ {
		roundStart := pocutil.NowMS()
		scalars := make([]gnarkbls12381fr.Element, size)
		offset := uint64(run*size + 1)
		for i := range scalars {
			scalars[i].SetUint64(offset + uint64(i))
		}
		result, err := new(gnarkbls12381.G1Affine).MultiExp(bases, scalars, ecc.MultiExpConfig{NbTasks: 1})
		if err != nil {
			return "", 0, err
		}
		digestBytes := make([]byte, 0, 96)
		digestBytes = append(digestBytes, bls12381FpMontLE(result.X)...)
		digestBytes = append(digestBytes, bls12381FpMontLE(result.Y)...)
		digests = append(digests, []byte(pocutil.SHA256Hex(digestBytes))...)
		roundDuration := pocutil.NowMS() - roundStart
		total += roundDuration
		pocutil.Logf("g1_msm_round_%d_ms = %.3f", run, roundDuration)
	}
	return pocutil.SHA256Hex(digests), total, nil
}

func decodeBN254Bases(fixture []byte, count int) []gnarkbn254.G1Affine {
	const coordinateBytes = 32
	const pointBytes = 96
	out := make([]gnarkbn254.G1Affine, count)
	for i := 0; i < count; i++ {
		base := i * pointBytes
		out[i] = gnarkbn254.G1Affine{
			X: readBN254FPMontLE(fixture[base : base+coordinateBytes]),
			Y: readBN254FPMontLE(fixture[base+coordinateBytes : base+2*coordinateBytes]),
		}
	}
	return out
}

func decodeBLS12381Bases(fixture []byte, count int) []gnarkbls12381.G1Affine {
	const coordinateBytes = 48
	const pointBytes = 144
	out := make([]gnarkbls12381.G1Affine, count)
	for i := 0; i < count; i++ {
		base := i * pointBytes
		out[i] = gnarkbls12381.G1Affine{
			X: readBLS12381FPMontLE(fixture[base : base+coordinateBytes]),
			Y: readBLS12381FPMontLE(fixture[base+coordinateBytes : base+2*coordinateBytes]),
		}
	}
	return out
}

func runBN254G2MSMBatch(size int, runs int, fixture []byte) (string, float64, error) {
	pointBytes := 192
	if len(fixture) < size*pointBytes {
		return "", 0, fmt.Errorf("g2 fixture has %d points, need %d", len(fixture)/pointBytes, size)
	}
	bases := decodeBN254G2Bases(fixture, size)
	digests := make([]byte, 0, runs*64)
	total := 0.0
	for run := 0; run < runs; run++ {
		roundStart := pocutil.NowMS()
		scalars := make([]gnarkbn254fr.Element, size)
		offset := uint64(run*size + 1)
		for i := range scalars {
			scalars[i].SetUint64(offset + uint64(i))
		}
		result, err := new(gnarkbn254.G2Affine).MultiExp(bases, scalars, ecc.MultiExpConfig{NbTasks: 1})
		if err != nil {
			return "", 0, err
		}
		digestBytes := make([]byte, 0, 128)
		digestBytes = append(digestBytes, bn254FpMontLE(result.X.A0)...)
		digestBytes = append(digestBytes, bn254FpMontLE(result.X.A1)...)
		digestBytes = append(digestBytes, bn254FpMontLE(result.Y.A0)...)
		digestBytes = append(digestBytes, bn254FpMontLE(result.Y.A1)...)
		digests = append(digests, []byte(pocutil.SHA256Hex(digestBytes))...)
		roundDuration := pocutil.NowMS() - roundStart
		total += roundDuration
		pocutil.Logf("g2_msm_round_%d_ms = %.3f", run, roundDuration)
	}
	return pocutil.SHA256Hex(digests), total, nil
}

func runBLS12381G2MSMBatch(size int, runs int, fixture []byte) (string, float64, error) {
	pointBytes := 288
	if len(fixture) < size*pointBytes {
		return "", 0, fmt.Errorf("g2 fixture has %d points, need %d", len(fixture)/pointBytes, size)
	}
	bases := decodeBLS12381G2Bases(fixture, size)
	digests := make([]byte, 0, runs*64)
	total := 0.0
	for run := 0; run < runs; run++ {
		roundStart := pocutil.NowMS()
		scalars := make([]gnarkbls12381fr.Element, size)
		offset := uint64(run*size + 1)
		for i := range scalars {
			scalars[i].SetUint64(offset + uint64(i))
		}
		result, err := new(gnarkbls12381.G2Affine).MultiExp(bases, scalars, ecc.MultiExpConfig{NbTasks: 1})
		if err != nil {
			return "", 0, err
		}
		digestBytes := make([]byte, 0, 192)
		digestBytes = append(digestBytes, bls12381FpMontLE(result.X.A0)...)
		digestBytes = append(digestBytes, bls12381FpMontLE(result.X.A1)...)
		digestBytes = append(digestBytes, bls12381FpMontLE(result.Y.A0)...)
		digestBytes = append(digestBytes, bls12381FpMontLE(result.Y.A1)...)
		digests = append(digests, []byte(pocutil.SHA256Hex(digestBytes))...)
		roundDuration := pocutil.NowMS() - roundStart
		total += roundDuration
		pocutil.Logf("g2_msm_round_%d_ms = %.3f", run, roundDuration)
	}
	return pocutil.SHA256Hex(digests), total, nil
}

func decodeBN254G2Bases(fixture []byte, count int) []gnarkbn254.G2Affine {
	const componentBytes = 32
	const pointBytes = 192
	out := make([]gnarkbn254.G2Affine, count)
	for i := 0; i < count; i++ {
		base := i * pointBytes
		out[i].X.A0 = readBN254FPMontLE(fixture[base : base+componentBytes])
		out[i].X.A1 = readBN254FPMontLE(fixture[base+componentBytes : base+2*componentBytes])
		out[i].Y.A0 = readBN254FPMontLE(fixture[base+2*componentBytes : base+3*componentBytes])
		out[i].Y.A1 = readBN254FPMontLE(fixture[base+3*componentBytes : base+4*componentBytes])
	}
	return out
}

func decodeBLS12381G2Bases(fixture []byte, count int) []gnarkbls12381.G2Affine {
	const componentBytes = 48
	const pointBytes = 288
	out := make([]gnarkbls12381.G2Affine, count)
	for i := 0; i < count; i++ {
		base := i * pointBytes
		out[i].X.A0 = readBLS12381FPMontLE(fixture[base : base+componentBytes])
		out[i].X.A1 = readBLS12381FPMontLE(fixture[base+componentBytes : base+2*componentBytes])
		out[i].Y.A0 = readBLS12381FPMontLE(fixture[base+2*componentBytes : base+3*componentBytes])
		out[i].Y.A1 = readBLS12381FPMontLE(fixture[base+3*componentBytes : base+4*componentBytes])
	}
	return out
}

func readBN254FPMontLE(src []byte) gnarkbn254fp.Element {
	var words [4]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(src[i*8 : (i+1)*8])
	}
	return gnarkbn254fp.Element(words)
}

func readBLS12381FPMontLE(src []byte) gnarkbls12381fp.Element {
	var words [6]uint64
	for i := range words {
		words[i] = binary.LittleEndian.Uint64(src[i*8 : (i+1)*8])
	}
	return gnarkbls12381fp.Element(words)
}

func bn254FpMontLE(v gnarkbn254fp.Element) []byte {
	out := make([]byte, 32)
	for i, word := range [4]uint64(v) {
		binary.LittleEndian.PutUint64(out[i*8:(i+1)*8], word)
	}
	return out
}

func bls12381FpMontLE(v gnarkbls12381fp.Element) []byte {
	out := make([]byte, 48)
	for i, word := range [6]uint64(v) {
		binary.LittleEndian.PutUint64(out[i*8:(i+1)*8], word)
	}
	return out
}
