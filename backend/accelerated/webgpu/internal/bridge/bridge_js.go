//go:build js && wasm

package bridge

import (
	"fmt"
	"syscall/js"
)

type Client struct {
	GlobalName  string
	ErrorPrefix string
}

func NewClient(globalName, errorPrefix string) Client {
	return Client{GlobalName: globalName, ErrorPrefix: errorPrefix}
}

func (c Client) getBridge() (js.Value, error) {
	bridge := js.Global().Get(c.GlobalName)
	if bridge.IsUndefined() || bridge.IsNull() {
		return js.Undefined(), fmt.Errorf("%s: %s bridge not found on global object", c.ErrorPrefix, c.GlobalName)
	}
	return bridge, nil
}

func (c Client) AwaitPromise(promise js.Value) (js.Value, error) {
	if promise.IsUndefined() || promise.IsNull() {
		return js.Undefined(), fmt.Errorf("%s: bridge returned empty promise", c.ErrorPrefix)
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
			err = c.JSError(args[0])
		} else {
			err = fmt.Errorf("%s: bridge promise rejected", c.ErrorPrefix)
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

func (c Client) JSError(v js.Value) error {
	if v.IsUndefined() || v.IsNull() {
		return fmt.Errorf("%s: unknown JS error", c.ErrorPrefix)
	}
	if message := v.Get("message"); message.Type() == js.TypeString {
		return fmt.Errorf("%s: %s", c.ErrorPrefix, message.String())
	}
	return fmt.Errorf("%s: %s", c.ErrorPrefix, v.String())
}

func (c Client) CallPromise(method string, args ...any) (js.Value, error) {
	bridge, err := c.getBridge()
	if err != nil {
		return js.Undefined(), err
	}
	fn := bridge.Get(method)
	if fn.Type() != js.TypeFunction {
		return js.Undefined(), fmt.Errorf("%s: bridge method %q is not available", c.ErrorPrefix, method)
	}
	return c.AwaitPromise(fn.Invoke(args...))
}

func JSUint8Array(src []byte) js.Value {
	out := js.Global().Get("Uint8Array").New(len(src))
	if len(src) > 0 {
		js.CopyBytesToJS(out, src)
	}
	return out
}

func GoBytes(prefix string, src js.Value) ([]byte, error) {
	if src.IsUndefined() || src.IsNull() {
		return nil, fmt.Errorf("%s: expected Uint8Array result, got empty value", prefix)
	}
	n := src.Get("byteLength")
	if n.Type() != js.TypeNumber {
		return nil, fmt.Errorf("%s: JS result does not expose byteLength", prefix)
	}
	out := make([]byte, n.Int())
	if len(out) > 0 {
		js.CopyBytesToGo(out, src)
	}
	return out, nil
}

func JSObject() js.Value {
	return js.Global().Get("Object").New()
}

func (c Client) Init(curve string) error {
	_, err := c.CallPromise("init", curve)
	return err
}

func (c Client) PrepareKey(curve string, payload js.Value) (string, error) {
	value, err := c.CallPromise("prepareKey", curve, payload)
	if err != nil {
		return "", err
	}
	handle := value.Get("handle")
	if handle.Type() != js.TypeString || handle.String() == "" {
		return "", fmt.Errorf("%s: bridge returned invalid key handle", c.ErrorPrefix)
	}
	return handle.String(), nil
}

func (c Client) MSMG1(handle, vectorName string, scalarsPacked []byte) ([]byte, error) {
	value, err := c.CallPromise("msmG1", handle, vectorName, JSUint8Array(scalarsPacked))
	if err != nil {
		return nil, err
	}
	return GoBytes(c.ErrorPrefix, value)
}

type MSMBatchResult struct {
	G1ABytes []byte
	G1BBytes []byte
	G1KBytes []byte
	G2BBytes []byte
}

func (c Client) MSMBatch(handle string, g1A, g1B, g1K []byte) (MSMBatchResult, error) {
	payload := JSObject()
	payload.Set("g1A", JSUint8Array(g1A))
	payload.Set("g1B", JSUint8Array(g1B))
	payload.Set("g1K", JSUint8Array(g1K))
	value, err := c.CallPromise("msmBatch", handle, payload)
	if err != nil {
		return MSMBatchResult{}, err
	}
	result := MSMBatchResult{}
	if result.G1ABytes, err = GoBytes(c.ErrorPrefix, value.Get("g1A")); err != nil {
		return MSMBatchResult{}, err
	}
	if result.G1BBytes, err = GoBytes(c.ErrorPrefix, value.Get("g1B")); err != nil {
		return MSMBatchResult{}, err
	}
	if result.G1KBytes, err = GoBytes(c.ErrorPrefix, value.Get("g1K")); err != nil {
		return MSMBatchResult{}, err
	}
	if result.G2BBytes, err = GoBytes(c.ErrorPrefix, value.Get("g2B")); err != nil {
		return MSMBatchResult{}, err
	}
	return result, nil
}

func (c Client) ComputeHZMSMG1(handle string, aPacked, bPacked, cPacked []byte) ([]byte, error) {
	value, err := c.CallPromise(
		"computeHZMSMG1",
		handle,
		JSUint8Array(aPacked),
		JSUint8Array(bPacked),
		JSUint8Array(cPacked),
	)
	if err != nil {
		return nil, err
	}
	return GoBytes(c.ErrorPrefix, value)
}

func (c Client) PrewarmQuotientDomain(curve string, domainSize int) error {
	_, err := c.CallPromise("prewarmQuotientDomain", curve, domainSize)
	return err
}
