//go:build js && wasm

package groth16

import (
	"syscall/js"

	webgpubridge "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/internal/bridge"
)

var bridgeClient = Groth16BridgeClient{Client: webgpubridge.NewClient("wnarkGroth16WebGPU", "webgpu groth16")}

type Groth16BridgeClient struct {
	webgpubridge.Client
}

type bridgeMSMBatchResult struct {
	G1ABytes []byte
	G1BBytes []byte
	G1KBytes []byte
	G2BBytes []byte
}

func jsUint8Array(src []byte) js.Value {
	return webgpubridge.JSUint8Array(src)
}

func jsObject() js.Value {
	return webgpubridge.JSObject()
}

func (c Groth16BridgeClient) MSMG1(handle, vectorName string, scalarsPacked []byte) ([]byte, error) {
	value, err := c.CallPromise("msmG1", handle, vectorName, webgpubridge.JSUint8Array(scalarsPacked))
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(c.ErrorPrefix, value)
}

func (c Groth16BridgeClient) MSMBatch(handle string, g1A, g1B, g1K []byte) (bridgeMSMBatchResult, error) {
	payload := webgpubridge.JSObject()
	payload.Set("g1A", webgpubridge.JSUint8Array(g1A))
	payload.Set("g1B", webgpubridge.JSUint8Array(g1B))
	payload.Set("g1K", webgpubridge.JSUint8Array(g1K))
	value, err := c.CallPromise("msmBatch", handle, payload)
	if err != nil {
		return bridgeMSMBatchResult{}, err
	}
	result := bridgeMSMBatchResult{}
	if result.G1ABytes, err = webgpubridge.GoBytes(c.ErrorPrefix, value.Get("g1A")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G1BBytes, err = webgpubridge.GoBytes(c.ErrorPrefix, value.Get("g1B")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G1KBytes, err = webgpubridge.GoBytes(c.ErrorPrefix, value.Get("g1K")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G2BBytes, err = webgpubridge.GoBytes(c.ErrorPrefix, value.Get("g2B")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	return result, nil
}

func (c Groth16BridgeClient) ComputeHZMSMG1(handle string, aPacked, bPacked, cPacked []byte) ([]byte, error) {
	value, err := c.CallPromise(
		"computeHZMSMG1",
		handle,
		webgpubridge.JSUint8Array(aPacked),
		webgpubridge.JSUint8Array(bPacked),
		webgpubridge.JSUint8Array(cPacked),
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(c.ErrorPrefix, value)
}

func (c Groth16BridgeClient) PrewarmQuotientDomain(curve string, domainSize int) error {
	_, err := c.CallPromise("prewarmQuotientDomain", curve, domainSize)
	return err
}
