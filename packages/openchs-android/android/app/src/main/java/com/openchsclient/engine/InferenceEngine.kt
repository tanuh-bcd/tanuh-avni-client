package com.openchsclient.engine

import com.openchsclient.preprocessing.ImagePreprocessor
import java.nio.ByteBuffer

/**
 * Abstract inference backend (~/.claude/plans/composed-tumbling-bachman.md).
 *
 * `EdgeModelModule` does not know about PyTorch, TFLite, ONNX, or any specific runtime —
 * it talks to this interface. The `engine` field in registry override JSON selects which
 * implementation handles a given model:
 *
 *   • "pytorch" → PyTorchEngine (this iteration)
 *
 * Future engines (TFLite, ONNX Runtime Mobile, ExecuTorch) drop in as additional
 * implementations behind this interface; the bridge code is untouched.
 *
 * ── Lifecycle ──────────────────────────────────────────────────────────────────────
 * `load` consumes a *plaintext* model buffer (post-AES-decrypt) and returns an opaque
 * `Handle`. The bridge holds on to the handle for the model's working lifetime. On
 * memory pressure the bridge calls `close(handle)` and the engine releases everything;
 * on the next inference the bridge re-loads via the cached load-args (self-heal path).
 */
interface InferenceEngine {

    /** Opaque per-engine state for one loaded model. */
    interface Handle {
        /**
         * Free all native resources. Must be idempotent — the bridge calls this on memory
         * pressure (`onTrimMemory`) and again on `onCatalystInstanceDestroy`.
         */
        fun close()
    }

    /**
     * Load a model from its plaintext bytes.
     *
     * `modelKey` is opaque metadata (used for log lines and, where the engine requires a
     * file path, to derive a per-model temp filename). `plaintext` is positioned at zero
     * and contains exactly one model.
     */
    fun load(modelKey: String, plaintext: ByteBuffer): Handle

    /**
     * Run inference on the preprocessor's output. Returns a flat `FloatArray` (caller
     * decodes via the configured `OutputDecoder`). The engine is responsible for shape
     * validation against the loaded model.
     */
    fun run(handle: Handle, input: ImagePreprocessor.Result): FloatArray

    /**
     * Run inference on a caller-supplied flat `FloatArray` + shape. Used by
     * `runInference(modelKey, number[])` JS callers that want to bypass the image pipeline.
     */
    fun run(handle: Handle, data: FloatArray, shape: LongArray): FloatArray
}
