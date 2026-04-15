//go:build js && wasm

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
	g1MSMSize := 1 << cfg.G1MSMLog
	g2MSMSize := 1 << cfg.G2MSMLog

	pocutil.Logf("=== Go Wasm -> TS/WebGPU (%s) ===", cfg.Curve)
	pocutil.Logf("ntt_size = %d", nttSize)
	pocutil.Logf("ntt_runs = %d", cfg.NTTRuns)
	pocutil.Logf("g1_msm_size = %d", g1MSMSize)
	pocutil.Logf("g1_msm_runs = %d", cfg.G1MSMRuns)
	pocutil.Logf("g2_msm_size = %d", g2MSMSize)
	pocutil.Logf("g2_msm_runs = %d", cfg.G2MSMRuns)

	bridge := js.Global().Get("curvegpuPocBridge")
	if bridge.IsUndefined() || bridge.IsNull() {
		return fmt.Errorf("missing curvegpuPocBridge")
	}

	overallStart := pocutil.NowMS()
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
	if fixturePoints := initValue.Get("g1FixturePoints"); fixturePoints.Type() == js.TypeNumber {
		pocutil.Logf("g1_fixture_points = %d", fixturePoints.Int())
	}
	if fixturePoints := initValue.Get("g2FixturePoints"); fixturePoints.Type() == js.TypeNumber {
		pocutil.Logf("g2_fixture_points = %d", fixturePoints.Int())
	}

	pocutil.SetStatus("Prewarming WebGPU runtime")
	prewarmValue, err := pocutil.Await(bridge.Call("prewarm", cfg.Curve, nttSize, g1MSMSize, g2MSMSize))
	if err != nil {
		return err
	}
	prewarmNTTDuration := prewarmValue.Get("nttMs").Float()
	prewarmG1MSMDuration := prewarmValue.Get("g1MsmMs").Float()
	prewarmG2MSMDuration := prewarmValue.Get("g2MsmMs").Float()
	prewarmDuration := prewarmValue.Get("totalMs").Float()
	pocutil.Logf("prewarm_ntt_ms = %.3f", prewarmNTTDuration)
	pocutil.Logf("prewarm_g1_msm_ms = %.3f", prewarmG1MSMDuration)
	pocutil.Logf("prewarm_g2_msm_ms = %.3f", prewarmG2MSMDuration)
	pocutil.Logf("prewarm_total_ms = %.3f", prewarmDuration)

	startupDuration := initDuration + prewarmDuration
	pocutil.Logf("startup_ms = %.3f", startupDuration)

	steadyStateStart := pocutil.NowMS()
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

	pocutil.SetStatus("Running G1 MSM batch")
	g1MSMDuration := 0.0
	g1MSMDigestInputs := make([]byte, 0, cfg.G1MSMRuns*64)
	for i := 0; i < cfg.G1MSMRuns; i++ {
		roundValue, err := pocutil.Await(bridge.Call("runG1Msm", cfg.Curve, g1MSMSize, i*g1MSMSize+1))
		if err != nil {
			return err
		}
		roundDuration := roundValue.Get("durationMs").Float()
		roundDigest := roundValue.Get("digestHex").String()
		g1MSMDuration += roundDuration
		g1MSMDigestInputs = append(g1MSMDigestInputs, []byte(roundDigest)...)
		pocutil.Logf("g1_msm_round_%d_ms = %.3f", i, roundDuration)
	}
	g1MSMDigest := pocutil.SHA256Hex(g1MSMDigestInputs)
	pocutil.Logf("g1_msm_total_ms = %.3f", g1MSMDuration)
	pocutil.Logf("g1_msm_avg_ms = %.3f", g1MSMDuration/float64(cfg.G1MSMRuns))
	pocutil.Logf("g1_msm_batch_digest = %s", g1MSMDigest)

	pocutil.SetStatus("Running G2 MSM batch")
	g2MSMDuration := 0.0
	g2MSMDigestInputs := make([]byte, 0, cfg.G2MSMRuns*64)
	for i := 0; i < cfg.G2MSMRuns; i++ {
		roundValue, err := pocutil.Await(bridge.Call("runG2Msm", cfg.Curve, g2MSMSize, i*g2MSMSize+1))
		if err != nil {
			return err
		}
		roundDuration := roundValue.Get("durationMs").Float()
		roundDigest := roundValue.Get("digestHex").String()
		g2MSMDuration += roundDuration
		g2MSMDigestInputs = append(g2MSMDigestInputs, []byte(roundDigest)...)
		pocutil.Logf("g2_msm_round_%d_ms = %.3f", i, roundDuration)
	}
	g2MSMDigest := pocutil.SHA256Hex(g2MSMDigestInputs)
	pocutil.Logf("g2_msm_total_ms = %.3f", g2MSMDuration)
	pocutil.Logf("g2_msm_avg_ms = %.3f", g2MSMDuration/float64(cfg.G2MSMRuns))
	pocutil.Logf("g2_msm_batch_digest = %s", g2MSMDigest)

	steadyStateDuration := pocutil.NowMS() - steadyStateStart
	overallDuration := pocutil.NowMS() - overallStart
	pocutil.Logf("steady_state_total_ms = %.3f", steadyStateDuration)
	pocutil.Logf("overall_total_ms = %.3f", overallDuration)

	pocutil.Complete(map[string]any{
		"impl":                       "webgpu-go",
		"curve":                      cfg.Curve,
		"init_duration_ms":           initDuration,
		"prewarm_duration_ms":        prewarmDuration,
		"prewarm_ntt_duration_ms":    prewarmNTTDuration,
		"prewarm_g1_msm_duration_ms": prewarmG1MSMDuration,
		"prewarm_g2_msm_duration_ms": prewarmG2MSMDuration,
		"startup_duration_ms":        startupDuration,
		"steady_state_duration_ms":   steadyStateDuration,
		"overall_duration_ms":        overallDuration,
		"ntt_size":                   nttSize,
		"ntt_runs":                   cfg.NTTRuns,
		"ntt_duration_ms":            nttDuration,
		"ntt_digest_hex":             nttDigest,
		"g1_msm_size":                g1MSMSize,
		"g1_msm_runs":                cfg.G1MSMRuns,
		"g1_msm_duration_ms":         g1MSMDuration,
		"g1_msm_digest_hex":          g1MSMDigest,
		"g2_msm_size":                g2MSMSize,
		"g2_msm_runs":                cfg.G2MSMRuns,
		"g2_msm_duration_ms":         g2MSMDuration,
		"g2_msm_digest_hex":          g2MSMDigest,
		"total_duration_ms":          overallDuration,
	})
	return nil
}
