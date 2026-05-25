//go:build js && wasm

package plonk

import (
	"syscall/js"

	webgpubridge "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/internal/bridge"
)

var bridgeClient = webgpubridge.NewClient("wnarkPlonkWebGPU", "webgpu plonk")

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

func bridgeMSMG1Slice(handle, vectorName string, start, count int, scalarsPacked []byte) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"msmG1",
		handle,
		vectorName,
		webgpubridge.JSUint8Array(scalarsPacked),
		start,
		count,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeMSMG1Batch(handle, vectorName string, start, termsPerInstance, instanceCount int, scalarsPacked []byte) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"msmG1Batch",
		handle,
		vectorName,
		webgpubridge.JSUint8Array(scalarsPacked),
		start,
		termsPerInstance,
		instanceCount,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeTransformQuotientCoset(curve string, valuesPacked, scalingPacked []byte, vectorCount, elementCount int) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"transformQuotientCoset",
		curve,
		webgpubridge.JSUint8Array(valuesPacked),
		webgpubridge.JSUint8Array(scalingPacked),
		vectorCount,
		elementCount,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeCanonicalizeQuotientFromCoset(curve string, valuesPacked []byte, elementCount int) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"canonicalizeQuotientFromCoset",
		curve,
		webgpubridge.JSUint8Array(valuesPacked),
		elementCount,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeCanonicalizeQuotientVectors(curve string, valuesPacked []byte, vectorCount, elementCount int, inputBitReversed, inverseCoset bool) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"canonicalizeQuotientVectors",
		curve,
		webgpubridge.JSUint8Array(valuesPacked),
		vectorCount,
		elementCount,
		inputBitReversed,
		inverseCoset,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeLagrangeQuotientVectors(curve string, valuesPacked []byte, vectorCount, elementCount int) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"lagrangeQuotientVectors",
		curve,
		webgpubridge.JSUint8Array(valuesPacked),
		vectorCount,
		elementCount,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeTransformAndEvaluateQuotientCoset(
	curve string,
	dynamicValuesPacked, scalingPacked, staticValuesPacked, twiddlesPacked, denominatorsPacked, blindsPacked, scalarsPacked []byte,
	elementCount, blindCoeffCount, commitmentCount, dynamicTransformCacheKey, staticMontCacheKey int,
) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"transformAndEvaluateQuotientCoset",
		curve,
		webgpubridge.JSUint8Array(dynamicValuesPacked),
		webgpubridge.JSUint8Array(scalingPacked),
		webgpubridge.JSUint8Array(staticValuesPacked),
		webgpubridge.JSUint8Array(twiddlesPacked),
		webgpubridge.JSUint8Array(denominatorsPacked),
		webgpubridge.JSUint8Array(blindsPacked),
		webgpubridge.JSUint8Array(scalarsPacked),
		elementCount,
		blindCoeffCount,
		commitmentCount,
		dynamicTransformCacheKey,
		staticMontCacheKey,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgePrewarmQuotientTransformDomain(curve string, elementCount int) error {
	_, err := bridgeClient.CallPromise("prewarmQuotientTransformDomain", curve, elementCount)
	return err
}

func bridgePrewarmQuotientCanonicalizeDomain(curve string, elementCount int) error {
	_, err := bridgeClient.CallPromise("prewarmQuotientCanonicalizeDomain", curve, elementCount)
	return err
}

func bridgePrewarmQuotientEvaluateKernel(curve string, commitmentCount int) error {
	_, err := bridgeClient.CallPromise("prewarmQuotientEvaluateKernel", curve, commitmentCount)
	return err
}
