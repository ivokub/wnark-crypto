//go:build js && wasm

package pocutil

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"syscall/js"

	"github.com/consensys/gnark-crypto/ecc"
	gnarkgroth16 "github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/ivokub/wnark-crypto/poc-gnark-groth16/common"
)

type Config struct {
	Curve     string
	SizeLog   int
	ProveRuns int
}

type ProveFunc func(constraint.ConstraintSystem, gnarkgroth16.ProvingKey, witness.Witness) (gnarkgroth16.Proof, error)

func ParseConfig() (Config, error) {
	cfg := js.Global().Get("__wnarkGroth16PocConfig")
	if cfg.IsUndefined() || cfg.IsNull() {
		return Config{}, errors.New("missing __wnarkGroth16PocConfig")
	}

	curve := cfg.Get("curve").String()
	if curve != "bn254" && curve != "bls12_381" {
		return Config{}, fmt.Errorf("unsupported curve %q", curve)
	}

	sizeLog := cfg.Get("sizeLog").Int()
	if sizeLog != 12 && sizeLog != 15 && sizeLog != 18 {
		return Config{}, fmt.Errorf("unsupported sizeLog %d", sizeLog)
	}

	proveRuns := cfg.Get("proveRuns").Int()
	if proveRuns <= 0 {
		return Config{}, fmt.Errorf("invalid proveRuns %d", proveRuns)
	}

	return Config{
		Curve:     curve,
		SizeLog:   sizeLog,
		ProveRuns: proveRuns,
	}, nil
}

func CurveID(curve string) (ecc.ID, error) {
	switch curve {
	case "bn254":
		return ecc.BN254, nil
	case "bls12_381":
		return ecc.BLS12_381, nil
	default:
		return ecc.UNKNOWN, fmt.Errorf("unsupported curve %q", curve)
	}
}

func FixtureDepth(sizeLog int) int {
	return 1 << sizeLog
}

func BuildWitness(curveID ecc.ID, depth int) (witness.Witness, witness.Witness, error) {
	field := curveID.ScalarField()
	out := common.ComputeOutput(field, 3, 5, depth)
	assignment := &common.MulAddChainCircuit{
		X:     3,
		Y:     5,
		Out:   out,
		Steps: depth,
	}

	fullWitness, err := frontend.NewWitness(assignment, field)
	if err != nil {
		return nil, nil, fmt.Errorf("full witness: %w", err)
	}
	publicWitness, err := fullWitness.Public()
	if err != nil {
		return nil, nil, fmt.Errorf("public witness: %w", err)
	}

	return fullWitness, publicWitness, nil
}

func RunHarness(implName string, cfg Config, loadFn func(ecc.ID) gnarkgroth16.ProvingKey, proveFn ProveFunc, prepareFn func(gnarkgroth16.ProvingKey) error) error {
	curveID, err := CurveID(cfg.Curve)
	if err != nil {
		return err
	}
	depth := FixtureDepth(cfg.SizeLog)

	Logf("=== %s (%s) ===", implName, cfg.Curve)
	Logf("fixture = 2^%d", cfg.SizeLog)
	Logf("depth = %d", depth)
	Logf("prove_runs = %d", cfg.ProveRuns)

	overallStart := NowMS()

	SetStatus("Loading fixture")
	fixtureStart := NowMS()
	ccs, pk, vk, err := LoadFixture(curveID, cfg.Curve, cfg.SizeLog, loadFn)
	if err != nil {
		return err
	}
	fixtureDuration := NowMS() - fixtureStart
	Logf("fixture_load_ms = %.3f", fixtureDuration)
	Logf("constraints = %d", ccs.GetNbConstraints())

	SetStatus("Building witness")
	witnessStart := NowMS()
	fullWitness, publicWitness, err := BuildWitness(curveID, depth)
	if err != nil {
		return err
	}
	witnessDuration := NowMS() - witnessStart
	Logf("witness_build_ms = %.3f", witnessDuration)

	prepareDuration := 0.0
	if prepareFn != nil {
		SetStatus("Preparing proving key")
		prepareStart := NowMS()
		if err := prepareFn(pk); err != nil {
			return fmt.Errorf("prepare proving key: %w", err)
		}
		prepareDuration = NowMS() - prepareStart
		Logf("prepare_ms = %.3f", prepareDuration)
	}

	startupDuration := fixtureDuration + witnessDuration + prepareDuration
	Logf("startup_ms = %.3f", startupDuration)

	proveDuration := 0.0
	serializeDuration := 0.0
	verifyDuration := 0.0
	proofSizeBytes := 0
	firstProofHash := ""

	steadyStateStart := NowMS()
	for i := 0; i < cfg.ProveRuns; i++ {
		SetStatus(fmt.Sprintf("Proving round %d/%d", i+1, cfg.ProveRuns))

		proveStart := NowMS()
		proof, err := proveFn(ccs, pk, fullWitness)
		if err != nil {
			return fmt.Errorf("prove round %d: %w", i, err)
		}
		roundProveDuration := NowMS() - proveStart
		proveDuration += roundProveDuration
		Logf("prove_round_%d_ms = %.3f", i, roundProveDuration)

		serializeStart := NowMS()
		proofBytes, err := SerializeProof(proof)
		if err != nil {
			return fmt.Errorf("serialize round %d: %w", i, err)
		}
		roundSerializeDuration := NowMS() - serializeStart

		verifyStart := NowMS()
		if err := VerifySerializedProof(curveID, proofBytes, vk, publicWitness); err != nil {
			return fmt.Errorf("verify round %d: %w", i, err)
		}
		roundVerifyDuration := NowMS() - verifyStart

		serializeDuration += roundSerializeDuration
		verifyDuration += roundVerifyDuration
		if proofSizeBytes == 0 {
			proofSizeBytes = len(proofBytes)
			firstProofHash = SHA256Hex(proofBytes)
			Logf("proof_size_bytes = %d", proofSizeBytes)
			Logf("proof_round_0_sha256 = %s", firstProofHash)
		}
		Logf("roundtrip_verify_round_%d = OK", i)
	}

	steadyStateDuration := NowMS() - steadyStateStart
	overallDuration := NowMS() - overallStart

	Logf("prove_total_ms = %.3f", proveDuration)
	Logf("prove_avg_ms = %.3f", proveDuration/float64(cfg.ProveRuns))
	Logf("serialize_total_ms = %.3f", serializeDuration)
	Logf("serialize_avg_ms = %.3f", serializeDuration/float64(cfg.ProveRuns))
	Logf("verify_total_ms = %.3f", verifyDuration)
	Logf("verify_avg_ms = %.3f", verifyDuration/float64(cfg.ProveRuns))
	Logf("steady_state_total_ms = %.3f", steadyStateDuration)
	Logf("overall_total_ms = %.3f", overallDuration)

	Complete(map[string]any{
		"impl":                       implName,
		"curve":                      cfg.Curve,
		"prove_runs":                 cfg.ProveRuns,
		"constraints":                ccs.GetNbConstraints(),
		"size_log":                   cfg.SizeLog,
		"depth_size":                 depth,
		"fixture_duration_ms":        fixtureDuration,
		"witness_duration_ms":        witnessDuration,
		"prepare_duration_ms":        prepareDuration,
		"startup_duration_ms":        startupDuration,
		"prove_duration_ms":          proveDuration,
		"serialize_duration_ms":      serializeDuration,
		"verify_duration_ms":         verifyDuration,
		"steady_state_duration_ms":   steadyStateDuration,
		"overall_duration_ms":        overallDuration,
		"proof_size_bytes":           proofSizeBytes,
		"proof_first_sha256":         firstProofHash,
		"roundtrip_verify_succeeded": true,
	})
	return nil
}

func FixtureBasePath(curve string, sizeLog int) string {
	return fmt.Sprintf("/poc-gnark-groth16/fixtures/%s/2pow%d", curve, sizeLog)
}

func LoadFixture(curveID ecc.ID, curve string, sizeLog int, pkFactory func(ecc.ID) gnarkgroth16.ProvingKey) (constraint.ConstraintSystem, gnarkgroth16.ProvingKey, gnarkgroth16.VerifyingKey, error) {
	base := FixtureBasePath(curve, sizeLog)
	ccsBytes, err := FetchBytes(base + "/ccs.bin")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fetch ccs: %w", err)
	}
	vkBytes, err := FetchBytes(base + "/vk.bin")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fetch vk: %w", err)
	}

	ccs := gnarkgroth16.NewCS(curveID)
	if _, err := ccs.ReadFrom(bytes.NewReader(ccsBytes)); err != nil {
		return nil, nil, nil, fmt.Errorf("read ccs: %w", err)
	}
	pk := pkFactory(curveID)
	pkDumpBytes, found, err := FetchBytesMaybe(base + "/pk.dump")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fetch pk dump: %w", err)
	}
	if found {
		if err := pk.ReadDump(bytes.NewReader(pkDumpBytes)); err != nil {
			return nil, nil, nil, fmt.Errorf("read pk dump: %w", err)
		}
	} else {
		pkBytes, err := FetchBytes(base + "/pk.bin")
		if err != nil {
			return nil, nil, nil, fmt.Errorf("fetch pk: %w", err)
		}
		if _, err := pk.ReadFrom(bytes.NewReader(pkBytes)); err != nil {
			return nil, nil, nil, fmt.Errorf("read pk: %w", err)
		}
	}
	vk := gnarkgroth16.NewVerifyingKey(curveID)
	if _, err := vk.ReadFrom(bytes.NewReader(vkBytes)); err != nil {
		return nil, nil, nil, fmt.Errorf("read vk: %w", err)
	}

	return ccs, pk, vk, nil
}

func SerializeProof(proof gnarkgroth16.Proof) ([]byte, error) {
	var buf bytes.Buffer
	if _, err := proof.WriteTo(&buf); err != nil {
		return nil, fmt.Errorf("serialize proof: %w", err)
	}
	return buf.Bytes(), nil
}

func VerifySerializedProof(curveID ecc.ID, proofBytes []byte, vk gnarkgroth16.VerifyingKey, publicWitness witness.Witness) error {
	roundTrip := gnarkgroth16.NewProof(curveID)
	if _, err := roundTrip.ReadFrom(bytes.NewReader(proofBytes)); err != nil {
		return fmt.Errorf("deserialize proof: %w", err)
	}
	if err := gnarkgroth16.Verify(roundTrip, vk, publicWitness); err != nil {
		return fmt.Errorf("verify proof: %w", err)
	}
	return nil
}

func Logf(format string, args ...any) {
	logger := js.Global().Get("__wnarkGroth16PocLog")
	if logger.IsUndefined() || logger.IsNull() {
		return
	}
	logger.Invoke(fmt.Sprintf(format, args...))
}

func SetStatus(text string) {
	status := js.Global().Get("__wnarkGroth16PocSetStatus")
	if status.IsUndefined() || status.IsNull() {
		return
	}
	status.Invoke(text)
}

func Complete(result map[string]any) {
	js.Global().Get("__wnarkGroth16PocComplete").Invoke(js.ValueOf(result))
}

func Fail(err error) {
	js.Global().Get("__wnarkGroth16PocFail").Invoke(err.Error())
}

func NowMS() float64 {
	return js.Global().Get("performance").Call("now").Float()
}

func FetchBytes(url string) ([]byte, error) {
	resp, err := Await(js.Global().Call("fetch", url))
	if err != nil {
		return nil, err
	}
	if !resp.Get("ok").Bool() {
		return nil, fmt.Errorf("fetch failed: %s", resp.Get("status").String())
	}
	arrayBuffer, err := Await(resp.Call("arrayBuffer"))
	if err != nil {
		return nil, err
	}
	uint8Array := js.Global().Get("Uint8Array").New(arrayBuffer)
	out := make([]byte, uint8Array.Get("length").Int())
	js.CopyBytesToGo(out, uint8Array)
	return out, nil
}

func FetchBytesMaybe(url string) ([]byte, bool, error) {
	resp, err := Await(js.Global().Call("fetch", url))
	if err != nil {
		return nil, false, err
	}
	if !resp.Get("ok").Bool() {
		status := resp.Get("status").Int()
		if status == 404 {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("fetch failed: %d", status)
	}
	arrayBuffer, err := Await(resp.Call("arrayBuffer"))
	if err != nil {
		return nil, false, err
	}
	uint8Array := js.Global().Get("Uint8Array").New(arrayBuffer)
	out := make([]byte, uint8Array.Get("length").Int())
	js.CopyBytesToGo(out, uint8Array)
	return out, true, nil
}

func Await(value js.Value) (js.Value, error) {
	if value.IsUndefined() || value.IsNull() {
		return value, nil
	}
	then := value.Get("then")
	if then.IsUndefined() || then.IsNull() || then.Type() != js.TypeFunction {
		return value, nil
	}

	type result struct {
		value js.Value
		err   error
	}
	done := make(chan result, 1)
	var once bool

	resolve := js.FuncOf(func(this js.Value, args []js.Value) any {
		if once {
			return nil
		}
		once = true
		var resolved js.Value
		if len(args) > 0 {
			resolved = args[0]
		}
		done <- result{value: resolved}
		return nil
	})
	reject := js.FuncOf(func(this js.Value, args []js.Value) any {
		if once {
			return nil
		}
		once = true
		message := "promise rejected"
		if len(args) > 0 {
			message = args[0].String()
		}
		done <- result{err: errors.New(message)}
		return nil
	})

	value.Call("then", resolve).Call("catch", reject)
	res := <-done
	resolve.Release()
	reject.Release()
	return res.value, res.err
}

func SHA256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
