package pocutil

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"syscall/js"
)

type Config struct {
	Curve           string
	NTTLog          int
	NTTRuns         int
	MSMLog          int
	MSMRuns         int
	FixtureBinPath  string
	CoordinateBytes int
	PointBytes      int
}

func ParseConfig() (Config, error) {
	cfg := js.Global().Get("__curvegpuPocConfig")
	if cfg.IsUndefined() || cfg.IsNull() {
		return Config{}, errors.New("missing __curvegpuPocConfig")
	}
	curve := cfg.Get("curve").String()
	if curve != "bn254" && curve != "bls12_381" {
		return Config{}, fmt.Errorf("unsupported curve %q", curve)
	}
	nttRuns := cfg.Get("nttRuns").Int()
	msmRuns := cfg.Get("msmRuns").Int()
	if nttRuns <= 0 {
		return Config{}, fmt.Errorf("invalid nttRuns %d", nttRuns)
	}
	if msmRuns <= 0 {
		return Config{}, fmt.Errorf("invalid msmRuns %d", msmRuns)
	}
	return Config{
		Curve:           curve,
		NTTLog:          cfg.Get("nttLog").Int(),
		NTTRuns:         nttRuns,
		MSMLog:          cfg.Get("msmLog").Int(),
		MSMRuns:         msmRuns,
		FixtureBinPath:  cfg.Get("fixtureBinPath").String(),
		CoordinateBytes: cfg.Get("coordinateBytes").Int(),
		PointBytes:      cfg.Get("pointBytes").Int(),
	}, nil
}

func Logf(format string, args ...any) {
	logger := js.Global().Get("__curvegpuPocLog")
	if logger.IsUndefined() || logger.IsNull() {
		return
	}
	logger.Invoke(fmt.Sprintf(format, args...))
}

func SetStatus(text string) {
	status := js.Global().Get("__curvegpuPocSetStatus")
	if status.IsUndefined() || status.IsNull() {
		return
	}
	status.Invoke(text)
}

func Complete(result map[string]any) {
	js.Global().Get("__curvegpuPocComplete").Invoke(js.ValueOf(result))
}

func Fail(err error) {
	js.Global().Get("__curvegpuPocFail").Invoke(err.Error())
}

func NowMS() float64 {
	return js.Global().Get("performance").Call("now").Float()
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

func LESequenceElement(index int, byteSize int) []byte {
	out := make([]byte, byteSize)
	value := uint64(index + 1)
	for i := 0; i < byteSize && value > 0; i++ {
		out[i] = byte(value)
		value >>= 8
	}
	return out
}

func ReverseCopy(src []byte) []byte {
	out := make([]byte, len(src))
	for i := range src {
		out[len(src)-1-i] = src[i]
	}
	return out
}

func SHA256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
