//go:build js && wasm

package groth16

import (
	"fmt"
	"syscall/js"
)

const bridgeGlobalName = "wnarkGroth16WebGPU"

func getBridge() (js.Value, error) {
	bridge := js.Global().Get(bridgeGlobalName)
	if bridge.IsUndefined() || bridge.IsNull() {
		return js.Undefined(), fmt.Errorf("webgpu groth16: %s bridge not found on global object", bridgeGlobalName)
	}
	return bridge, nil
}

func awaitPromise(promise js.Value) (js.Value, error) {
	if promise.IsUndefined() || promise.IsNull() {
		return js.Undefined(), fmt.Errorf("webgpu groth16: bridge returned empty promise")
	}

	type result struct {
		value js.Value
		err   error
	}
	ch := make(chan result, 1)

	resolve := js.FuncOf(func(this js.Value, args []js.Value) any {
		value := js.Undefined()
		if len(args) > 0 {
			value = args[0]
		}
		ch <- result{value: value}
		return nil
	})
	reject := js.FuncOf(func(this js.Value, args []js.Value) any {
		var err error
		if len(args) > 0 {
			err = jsError(args[0])
		} else {
			err = fmt.Errorf("webgpu groth16: bridge promise rejected")
		}
		ch <- result{err: err}
		return nil
	})
	defer resolve.Release()
	defer reject.Release()

	promise.Call("then", resolve, reject)
	out := <-ch
	return out.value, out.err
}

func jsError(v js.Value) error {
	if v.IsUndefined() || v.IsNull() {
		return fmt.Errorf("webgpu groth16: unknown JS error")
	}
	if message := v.Get("message"); message.Type() == js.TypeString {
		return fmt.Errorf("webgpu groth16: %s", message.String())
	}
	return fmt.Errorf("webgpu groth16: %s", v.String())
}

func callPromise(method string, args ...any) (js.Value, error) {
	bridge, err := getBridge()
	if err != nil {
		return js.Undefined(), err
	}
	fn := bridge.Get(method)
	if fn.Type() != js.TypeFunction {
		return js.Undefined(), fmt.Errorf("webgpu groth16: bridge method %q is not available", method)
	}
	return awaitPromise(fn.Invoke(args...))
}

func jsUint8Array(src []byte) js.Value {
	out := js.Global().Get("Uint8Array").New(len(src))
	if len(src) > 0 {
		js.CopyBytesToJS(out, src)
	}
	return out
}

func goBytes(src js.Value) ([]byte, error) {
	if src.IsUndefined() || src.IsNull() {
		return nil, fmt.Errorf("webgpu groth16: expected Uint8Array result, got empty value")
	}
	n := src.Get("byteLength")
	if n.Type() != js.TypeNumber {
		return nil, fmt.Errorf("webgpu groth16: JS result does not expose byteLength")
	}
	out := make([]byte, n.Int())
	if len(out) > 0 {
		js.CopyBytesToGo(out, src)
	}
	return out, nil
}

func jsObject() js.Value {
	return js.Global().Get("Object").New()
}

func bridgeInit(curve string) error {
	_, err := callPromise("init", curve)
	return err
}

func bridgePrepareKey(curve string, payload js.Value) (string, error) {
	value, err := callPromise("prepareKey", curve, payload)
	if err != nil {
		return "", err
	}
	handle := value.Get("handle")
	if handle.Type() != js.TypeString || handle.String() == "" {
		return "", fmt.Errorf("webgpu groth16: bridge returned invalid key handle")
	}
	return handle.String(), nil
}

func bridgeMSMG1(handle, vectorName string, scalarsPacked []byte) ([]byte, error) {
	value, err := callPromise("msmG1", handle, vectorName, jsUint8Array(scalarsPacked))
	if err != nil {
		return nil, err
	}
	return goBytes(value)
}

type bridgeMSMBatchResult struct {
	G1ABytes []byte
	G1BBytes []byte
	G1KBytes []byte
	G2BBytes []byte
}

func bridgeMSMBatch(handle string, g1A, g1B, g1K []byte) (bridgeMSMBatchResult, error) {
	payload := jsObject()
	payload.Set("g1A", jsUint8Array(g1A))
	payload.Set("g1B", jsUint8Array(g1B))
	payload.Set("g1K", jsUint8Array(g1K))
	value, err := callPromise("msmBatch", handle, payload)
	if err != nil {
		return bridgeMSMBatchResult{}, err
	}
	result := bridgeMSMBatchResult{}
	if result.G1ABytes, err = goBytes(value.Get("g1A")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G1BBytes, err = goBytes(value.Get("g1B")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G1KBytes, err = goBytes(value.Get("g1K")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G2BBytes, err = goBytes(value.Get("g2B")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	return result, nil
}

func bridgeComputeHZMSMG1(handle string, aPacked, bPacked, cPacked []byte) ([]byte, error) {
	value, err := callPromise(
		"computeHZMSMG1",
		handle,
		jsUint8Array(aPacked),
		jsUint8Array(bPacked),
		jsUint8Array(cPacked),
	)
	if err != nil {
		return nil, err
	}
	return goBytes(value)
}

func bridgePrewarmQuotientDomain(curve string, domainSize int) error {
	_, err := callPromise("prewarmQuotientDomain", curve, domainSize)
	return err
}
