//go:build js && wasm

package groth16

import (
	"syscall/js"

	webgpubridge "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/internal/bridge"
)

var bridgeClient = webgpubridge.NewClient("wnarkGroth16WebGPU", "webgpu groth16")

type bridgeMSMBatchResult = webgpubridge.MSMBatchResult

func jsUint8Array(src []byte) js.Value {
	return webgpubridge.JSUint8Array(src)
}

func jsObject() js.Value {
	return webgpubridge.JSObject()
}

func bridgeInit(curve string) error {
	return bridgeClient.Init(curve)
}

func bridgePrepareKey(curve string, payload js.Value) (string, error) {
	return bridgeClient.PrepareKey(curve, payload)
}

func bridgeMSMG1(handle, vectorName string, scalarsPacked []byte) ([]byte, error) {
	return bridgeClient.MSMG1(handle, vectorName, scalarsPacked)
}

func bridgeMSMBatch(handle string, g1A, g1B, g1K []byte) (bridgeMSMBatchResult, error) {
	return bridgeClient.MSMBatch(handle, g1A, g1B, g1K)
}

func bridgeComputeHZMSMG1(handle string, aPacked, bPacked, cPacked []byte) ([]byte, error) {
	return bridgeClient.ComputeHZMSMG1(handle, aPacked, bPacked, cPacked)
}

func bridgePrewarmQuotientDomain(curve string, domainSize int) error {
	return bridgeClient.PrewarmQuotientDomain(curve, domainSize)
}
