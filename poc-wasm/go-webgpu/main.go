package main

import (
	"fmt"
	"syscall/js"

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

	pocutil.Logf("=== Go Wasm -> TS/WebGPU (%s) ===", cfg.Curve)
	pocutil.Logf("ntt_size = %d", nttSize)
	pocutil.Logf("ntt_runs = %d", cfg.NTTRuns)
	pocutil.Logf("msm_size = %d", msmSize)
	pocutil.Logf("msm_runs = %d", cfg.MSMRuns)

	bridge := js.Global().Get("curvegpuPocBridge")
	if bridge.IsUndefined() || bridge.IsNull() {
		return fmt.Errorf("missing curvegpuPocBridge")
	}

	totalStart := pocutil.NowMS()
	pocutil.SetStatus("Initializing WebGPU bridge")
	initStart := pocutil.NowMS()
	initValue, err := pocutil.Await(bridge.Call("init", cfg.Curve))
	if err != nil {
		return err
	}
	initDuration := pocutil.NowMS() - initStart
	pocutil.Logf("init_ms = %.3f", initDuration)
	if vendor := initValue.Get("adapter").Get("vendor").String(); vendor != "" {
		pocutil.Logf("adapter.vendor = %s", vendor)
	}
	if arch := initValue.Get("adapter").Get("architecture").String(); arch != "" {
		pocutil.Logf("adapter.architecture = %s", arch)
	}
	if fixturePoints := initValue.Get("fixturePoints"); fixturePoints.Type() == js.TypeNumber {
		pocutil.Logf("fixture_points = %d", fixturePoints.Int())
	}

	pocutil.SetStatus("Running NTT batch")
	nttDuration := 0.0
	nttDigestInputs := make([]byte, 0, cfg.NTTRuns*64)
	for i := 0; i < cfg.NTTRuns; i++ {
		roundValue, err := pocutil.Await(bridge.Call("runNtt", cfg.Curve, nttSize, i*nttSize+1))
		if err != nil {
			return err
		}
		roundDuration := roundValue.Get("durationMs").Float()
		roundDigest := roundValue.Get("digestHex").String()
		nttDuration += roundDuration
		nttDigestInputs = append(nttDigestInputs, []byte(roundDigest)...)
		pocutil.Logf("ntt_round_%d_ms = %.3f", i, roundDuration)
	}
	nttDigest := pocutil.SHA256Hex(nttDigestInputs)
	pocutil.Logf("ntt_total_ms = %.3f", nttDuration)
	pocutil.Logf("ntt_avg_ms = %.3f", nttDuration/float64(cfg.NTTRuns))
	pocutil.Logf("ntt_batch_digest = %s", nttDigest)

	pocutil.SetStatus("Running MSM batch")
	msmDuration := 0.0
	msmDigestInputs := make([]byte, 0, cfg.MSMRuns*64)
	for i := 0; i < cfg.MSMRuns; i++ {
		roundValue, err := pocutil.Await(bridge.Call("runMsm", cfg.Curve, msmSize, i*msmSize+1))
		if err != nil {
			return err
		}
		roundDuration := roundValue.Get("durationMs").Float()
		roundDigest := roundValue.Get("digestHex").String()
		msmDuration += roundDuration
		msmDigestInputs = append(msmDigestInputs, []byte(roundDigest)...)
		pocutil.Logf("msm_round_%d_ms = %.3f", i, roundDuration)
	}
	msmDigest := pocutil.SHA256Hex(msmDigestInputs)
	pocutil.Logf("msm_total_ms = %.3f", msmDuration)
	pocutil.Logf("msm_avg_ms = %.3f", msmDuration/float64(cfg.MSMRuns))
	pocutil.Logf("msm_batch_digest = %s", msmDigest)

	totalDuration := pocutil.NowMS() - totalStart
	pocutil.Logf("total_ms = %.3f", totalDuration)

	pocutil.Complete(map[string]any{
		"impl":              "webgpu-go",
		"curve":             cfg.Curve,
		"init_duration_ms":  initDuration,
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
