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
	msmSize := 1 << cfg.MSMLog

	pocutil.Logf("=== Go Wasm -> gnark-crypto (%s) ===", cfg.Curve)
	pocutil.Logf("ntt_size = %d", nttSize)
	pocutil.Logf("ntt_runs = %d", cfg.NTTRuns)
	pocutil.Logf("msm_size = %d", msmSize)
	pocutil.Logf("msm_runs = %d", cfg.MSMRuns)

	var fixtureBytes []byte
	fixtureLoadMs := 0.0
	if cfg.FixtureBinPath != "" {
		pocutil.SetStatus("Loading fixture")
		start := pocutil.NowMS()
		fixtureBytes, err = pocutil.FetchBytes(cfg.FixtureBinPath)
		if err != nil {
			return err
		}
		fixtureLoadMs = pocutil.NowMS() - start
		pocutil.Logf("fixture_load_ms = %.3f", fixtureLoadMs)
	}

	totalStart := pocutil.NowMS()
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

	pocutil.SetStatus("Running MSM batch")
	var msmDigest string
	var msmDuration float64
	switch cfg.Curve {
	case "bn254":
		msmDigest, msmDuration, err = runBN254MSMBatch(msmSize, cfg.MSMRuns, fixtureBytes)
	case "bls12_381":
		msmDigest, msmDuration, err = runBLS12381MSMBatch(msmSize, cfg.MSMRuns, fixtureBytes)
	default:
		err = fmt.Errorf("unsupported curve %q", cfg.Curve)
	}
	if err != nil {
		return err
	}
	pocutil.Logf("msm_total_ms = %.3f", msmDuration)
	pocutil.Logf("msm_avg_ms = %.3f", msmDuration/float64(cfg.MSMRuns))
	pocutil.Logf("msm_batch_digest = %s", msmDigest)

	totalDuration := pocutil.NowMS() - totalStart
	pocutil.Logf("total_ms = %.3f", totalDuration)

	pocutil.Complete(map[string]any{
		"impl":              "gnark-go",
		"curve":             cfg.Curve,
		"fixture_load_ms":   fixtureLoadMs,
		"ntt_size":          nttSize,
		"ntt_runs":          cfg.NTTRuns,
		"ntt_duration_ms":   nttDuration,
		"ntt_digest_hex":    nttDigest,
		"msm_size":          msmSize,
		"msm_runs":          cfg.MSMRuns,
		"msm_duration_ms":   msmDuration,
		"msm_digest_hex":    msmDigest,
		"total_duration_ms": totalDuration,
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

func runBN254MSMBatch(size int, runs int, fixture []byte) (string, float64, error) {
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
		pocutil.Logf("msm_round_%d_ms = %.3f", run, roundDuration)
	}
	return pocutil.SHA256Hex(digests), total, nil
}

func runBLS12381MSMBatch(size int, runs int, fixture []byte) (string, float64, error) {
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
		pocutil.Logf("msm_round_%d_ms = %.3f", run, roundDuration)
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
