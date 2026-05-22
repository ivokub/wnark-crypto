//go:build js && wasm

package wasmruntime

import (
	"bytes"
	"fmt"
	"io"
	"syscall/js"

	"github.com/consensys/gnark-crypto/ecc"
	gnarkplonk "github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
)

type ProveFunc func(constraint.ConstraintSystem, gnarkplonk.ProvingKey, witness.Witness) (gnarkplonk.Proof, error)
type PrepareFunc func(gnarkplonk.ProvingKey) error

type Config struct {
	GlobalName string
	PKFactory  func(ecc.ID) gnarkplonk.ProvingKey
	Prepare    PrepareFunc
	Prove      ProveFunc
}

type Runtime struct {
	cfg     Config
	next    uint64
	funcs   []js.Func
	ccs     map[string]ccsEntry
	pks     map[string]pkEntry
	vks     map[string]vkEntry
	handles map[string]string
}

type ccsEntry struct {
	curve ecc.ID
	value constraint.ConstraintSystem
}

type pkEntry struct {
	curve    ecc.ID
	value    gnarkplonk.ProvingKey
	prepared bool
}

type vkEntry struct {
	curve ecc.ID
	value gnarkplonk.VerifyingKey
}

func Install(cfg Config) error {
	if cfg.GlobalName == "" {
		return fmt.Errorf("missing global name")
	}
	if cfg.PKFactory == nil {
		return fmt.Errorf("missing proving key factory")
	}
	if cfg.Prove == nil {
		return fmt.Errorf("missing prove function")
	}

	r := &Runtime{
		cfg:     cfg,
		ccs:     make(map[string]ccsEntry),
		pks:     make(map[string]pkEntry),
		vks:     make(map[string]vkEntry),
		handles: make(map[string]string),
	}
	js.Global().Set(cfg.GlobalName, r.object())
	select {}
}

func (r *Runtime) object() js.Value {
	obj := js.Global().Get("Object").New()
	r.setMethod(obj, "readConstraintSystem", r.readConstraintSystem)
	r.setMethod(obj, "readProvingKey", r.readProvingKey)
	r.setMethod(obj, "readVerificationKey", r.readVerificationKey)
	r.setMethod(obj, "prepareProvingKey", r.prepareProvingKey)
	r.setMethod(obj, "prove", r.prove)
	r.setMethod(obj, "verify", r.verify)
	r.setMethod(obj, "release", r.release)
	return obj
}

func (r *Runtime) setMethod(obj js.Value, name string, fn func([]js.Value) (js.Value, error)) {
	callback := js.FuncOf(func(this js.Value, args []js.Value) any {
		return promise(func() (js.Value, error) {
			return fn(args)
		})
	})
	r.funcs = append(r.funcs, callback)
	obj.Set(name, callback)
}

func promise(fn func() (js.Value, error)) js.Value {
	executor := js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve := args[0]
		reject := args[1]
		go func() {
			value, err := fn()
			if err != nil {
				reject.Invoke(js.Global().Get("Error").New(err.Error()))
				return
			}
			resolve.Invoke(value)
		}()
		return nil
	})
	p := js.Global().Get("Promise").New(executor)
	executor.Release()
	return p
}

func (r *Runtime) readConstraintSystem(args []js.Value) (js.Value, error) {
	curveID, err := curveIDFromArg(args, 0)
	if err != nil {
		return js.Undefined(), err
	}
	data, err := bytesFromArg(args, 1)
	if err != nil {
		return js.Undefined(), err
	}
	ccs := gnarkplonk.NewCS(curveID)
	if _, err := ccs.ReadFrom(bytes.NewReader(data)); err != nil {
		return js.Undefined(), fmt.Errorf("read ccs: %w", err)
	}
	handle := r.store("ccs")
	r.ccs[handle] = ccsEntry{curve: curveID, value: ccs}

	out := js.Global().Get("Object").New()
	out.Set("handle", handle)
	out.Set("constraints", ccs.GetNbConstraints())
	return out, nil
}

func (r *Runtime) readProvingKey(args []js.Value) (js.Value, error) {
	curveID, err := curveIDFromArg(args, 0)
	if err != nil {
		return js.Undefined(), err
	}
	data, err := bytesFromArg(args, 1)
	if err != nil {
		return js.Undefined(), err
	}
	format := "serialized"
	if len(args) > 2 && args[2].Type() == js.TypeString {
		format = args[2].String()
	}
	pk := r.cfg.PKFactory(curveID)
	switch format {
	case "serialized":
		if _, err := pk.ReadFrom(bytes.NewReader(data)); err != nil {
			return js.Undefined(), fmt.Errorf("read pk: %w", err)
		}
	case "unsafe":
		if _, err := pk.UnsafeReadFrom(bytes.NewReader(data)); err != nil {
			return js.Undefined(), fmt.Errorf("read pk unsafe: %w", err)
		}
	default:
		return js.Undefined(), fmt.Errorf("unsupported proving key format %q", format)
	}
	handle := r.store("pk")
	r.pks[handle] = pkEntry{curve: curveID, value: pk}
	return handleObject(handle), nil
}

func (r *Runtime) readVerificationKey(args []js.Value) (js.Value, error) {
	curveID, err := curveIDFromArg(args, 0)
	if err != nil {
		return js.Undefined(), err
	}
	data, err := bytesFromArg(args, 1)
	if err != nil {
		return js.Undefined(), err
	}
	vk := gnarkplonk.NewVerifyingKey(curveID)
	if _, err := vk.ReadFrom(bytes.NewReader(data)); err != nil {
		return js.Undefined(), fmt.Errorf("read vk: %w", err)
	}
	handle := r.store("vk")
	r.vks[handle] = vkEntry{curve: curveID, value: vk}
	return handleObject(handle), nil
}

func (r *Runtime) prepareProvingKey(args []js.Value) (js.Value, error) {
	handle, pk, err := r.pkFromArg(args, 0)
	if err != nil {
		return js.Undefined(), err
	}
	if err := r.ensurePrepared(handle, pk); err != nil {
		return js.Undefined(), err
	}
	return js.Undefined(), nil
}

func (r *Runtime) prove(args []js.Value) (js.Value, error) {
	ccs, err := r.ccsFromArg(args, 0)
	if err != nil {
		return js.Undefined(), err
	}
	pkHandle, pk, err := r.pkFromArg(args, 1)
	if err != nil {
		return js.Undefined(), err
	}
	if ccs.curve != pk.curve {
		return js.Undefined(), fmt.Errorf("ccs and proving key curves do not match")
	}
	witnessBytes, err := bytesFromArg(args, 2)
	if err != nil {
		return js.Undefined(), err
	}
	fullWitness, err := readWitness(ccs.curve, witnessBytes)
	if err != nil {
		return js.Undefined(), fmt.Errorf("read witness: %w", err)
	}
	if err := r.ensurePrepared(pkHandle, pk); err != nil {
		return js.Undefined(), err
	}
	proof, err := r.cfg.Prove(ccs.value, pk.value, fullWitness)
	if err != nil {
		return js.Undefined(), fmt.Errorf("prove: %w", err)
	}
	proofBytes, err := writeToBytes(proof)
	if err != nil {
		return js.Undefined(), fmt.Errorf("serialize proof: %w", err)
	}
	return jsBytes(proofBytes), nil
}

func (r *Runtime) verify(args []js.Value) (js.Value, error) {
	proofBytes, err := bytesFromArg(args, 0)
	if err != nil {
		return js.Undefined(), err
	}
	vk, err := r.vkFromArg(args, 1)
	if err != nil {
		return js.Undefined(), err
	}
	publicWitnessBytes, err := bytesFromArg(args, 2)
	if err != nil {
		return js.Undefined(), err
	}
	proof := gnarkplonk.NewProof(vk.curve)
	if _, err := proof.ReadFrom(bytes.NewReader(proofBytes)); err != nil {
		return js.Undefined(), fmt.Errorf("read proof: %w", err)
	}
	publicWitness, err := readWitness(vk.curve, publicWitnessBytes)
	if err != nil {
		return js.Undefined(), fmt.Errorf("read public witness: %w", err)
	}
	if err := gnarkplonk.Verify(proof, vk.value, publicWitness); err != nil {
		return js.ValueOf(false), nil
	}
	return js.ValueOf(true), nil
}

func (r *Runtime) release(args []js.Value) (js.Value, error) {
	if len(args) < 1 || args[0].Type() != js.TypeString {
		return js.Undefined(), fmt.Errorf("missing handle")
	}
	handle := args[0].String()
	switch r.handles[handle] {
	case "ccs":
		delete(r.ccs, handle)
	case "pk":
		delete(r.pks, handle)
	case "vk":
		delete(r.vks, handle)
	}
	delete(r.handles, handle)
	return js.Undefined(), nil
}

func (r *Runtime) ensurePrepared(handle string, pk pkEntry) error {
	if pk.prepared || r.cfg.Prepare == nil {
		return nil
	}
	if err := r.cfg.Prepare(pk.value); err != nil {
		return fmt.Errorf("prepare pk: %w", err)
	}
	pk.prepared = true
	r.pks[handle] = pk
	return nil
}

func (r *Runtime) ccsFromArg(args []js.Value, index int) (ccsEntry, error) {
	handle, err := handleFromArg(args, index)
	if err != nil {
		return ccsEntry{}, err
	}
	entry, ok := r.ccs[handle]
	if !ok {
		return ccsEntry{}, fmt.Errorf("unknown ccs handle %q", handle)
	}
	return entry, nil
}

func (r *Runtime) pkFromArg(args []js.Value, index int) (string, pkEntry, error) {
	handle, err := handleFromArg(args, index)
	if err != nil {
		return "", pkEntry{}, err
	}
	entry, ok := r.pks[handle]
	if !ok {
		return "", pkEntry{}, fmt.Errorf("unknown proving key handle %q", handle)
	}
	return handle, entry, nil
}

func (r *Runtime) vkFromArg(args []js.Value, index int) (vkEntry, error) {
	handle, err := handleFromArg(args, index)
	if err != nil {
		return vkEntry{}, err
	}
	entry, ok := r.vks[handle]
	if !ok {
		return vkEntry{}, fmt.Errorf("unknown verification key handle %q", handle)
	}
	return entry, nil
}

func (r *Runtime) store(kind string) string {
	r.next++
	handle := fmt.Sprintf("%s:%d", kind, r.next)
	r.handles[handle] = kind
	return handle
}

func handleObject(handle string) js.Value {
	out := js.Global().Get("Object").New()
	out.Set("handle", handle)
	return out
}

func curveIDFromArg(args []js.Value, index int) (ecc.ID, error) {
	if len(args) <= index || args[index].Type() != js.TypeString {
		return ecc.UNKNOWN, fmt.Errorf("missing curve")
	}
	switch args[index].String() {
	case "bn254":
		return ecc.BN254, nil
	default:
		return ecc.UNKNOWN, fmt.Errorf("unsupported plonk curve %q", args[index].String())
	}
}

func handleFromArg(args []js.Value, index int) (string, error) {
	if len(args) <= index || args[index].Type() != js.TypeString {
		return "", fmt.Errorf("missing handle")
	}
	return args[index].String(), nil
}

func bytesFromArg(args []js.Value, index int) ([]byte, error) {
	if len(args) <= index {
		return nil, fmt.Errorf("missing bytes argument")
	}
	src := args[index]
	n := src.Get("byteLength")
	if n.Type() != js.TypeNumber {
		return nil, fmt.Errorf("expected Uint8Array")
	}
	out := make([]byte, n.Int())
	if len(out) > 0 {
		js.CopyBytesToGo(out, src)
	}
	return out, nil
}

func jsBytes(src []byte) js.Value {
	out := js.Global().Get("Uint8Array").New(len(src))
	if len(src) > 0 {
		js.CopyBytesToJS(out, src)
	}
	return out
}

func readWitness(curveID ecc.ID, data []byte) (witness.Witness, error) {
	w, err := witness.New(curveID.ScalarField())
	if err != nil {
		return nil, err
	}
	if _, err := w.ReadFrom(bytes.NewReader(data)); err != nil {
		return nil, err
	}
	return w, nil
}

func writeToBytes(value interface {
	WriteTo(io.Writer) (int64, error)
}) ([]byte, error) {
	var buf bytes.Buffer
	if _, err := value.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
