package com.openchsclient.engine

import android.content.Context
import android.util.Log
import com.openchsclient.preprocessing.ImagePreprocessor
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

/**
 * PyTorch Mobile inference backend (~/.claude/plans/composed-tumbling-bachman.md).
 *
 * Uses LibTorch's Java bindings (`org.pytorch:pytorch_android:1.13.1`). This version is
 * pinned exactly to the TANUH PoC at `~/IdeaProjects/aiapp` because clinical accuracy was
 * validated against this exact runtime — upgrading would force re-validation.
 *
 * ── Decrypt-then-load tradeoff ─────────────────────────────────────────────────────
 * `org.pytorch.Module.load` requires a **file path**, not a buffer. We therefore have to
 * land the plaintext on disk briefly:
 *
 *   1. Write the decrypted plaintext into the app's private `filesDir/<modelKey>.pt` with
 *      mode 0600 (owner read/write only — Android's `MODE_PRIVATE` enforces this).
 *   2. `Module.load(path)`. The runtime mmap's the file; the in-memory model lives in
 *      LibTorch's native heap from this point on.
 *   3. Delete the file in a `finally` block.
 *
 * Plaintext exists on disk only inside this brief window. This is consistent with the
 * already-documented threat model in `tools/edge-model/README.md`: the AES key ships in the
 * APK, so the encryption is obfuscation rather than full IP protection. A determined
 * reverser can recover the plaintext via much easier paths than racing this decrypt window.
 *
 * Future hardening: a custom JNI shim that calls `torch::jit::load(istream)` directly would
 * keep plaintext entirely off-disk. ~1-2 days of native work, deferred until needed.
 */
class PyTorchEngine(private val context: Context) : InferenceEngine {

    companion object { private const val TAG = "PyTorchEngine" }

    /**
     * Per-model state. We keep the loaded `Module` plus the path of the temp file so we can
     * verify it's been deleted. (The file *is* deleted post-load; this field is diagnostic.)
     */
    private class PyTorchHandle(val module: Module) : InferenceEngine.Handle {
        @Volatile private var closed = false
        override fun close() {
            if (closed) return
            closed = true
            try { module.destroy() } catch (e: Exception) { Log.w(TAG, "Module.destroy: ${e.message}") }
        }
    }

    override fun load(modelKey: String, plaintext: ByteBuffer): InferenceEngine.Handle {
        // Sanitise the key — it becomes a filename. Allow alnum, dash, underscore, dot.
        val safeKey = modelKey.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val tempFile = File(context.filesDir, "$safeKey.pt.tmp")
        try {
            // MODE_PRIVATE on filesDir + FileOutputStream → 0600 by default on modern Android.
            FileOutputStream(tempFile).use { fos ->
                val channel = fos.channel
                plaintext.rewind()
                channel.write(plaintext)
            }
            // Defensive — older Android variants don't always honour the mode. No-op on success.
            try { tempFile.setReadable(false, false); tempFile.setReadable(true, true) } catch (_: Exception) {}

            Log.d(TAG, "load($modelKey): plaintext=${tempFile.length()} bytes at ${tempFile.absolutePath}")
            val module = Module.load(tempFile.absolutePath)
            return PyTorchHandle(module)
        } finally {
            // Whether load succeeded or threw, the plaintext file must not linger.
            if (tempFile.exists()) {
                val deleted = tempFile.delete()
                if (!deleted) Log.w(TAG, "load($modelKey): failed to delete temp file ${tempFile.absolutePath}")
            }
        }
    }

    override fun run(handle: InferenceEngine.Handle, input: ImagePreprocessor.Result): FloatArray =
        run(handle, input.data, input.shape)

    override fun run(handle: InferenceEngine.Handle, data: FloatArray, shape: LongArray): FloatArray {
        val module = (handle as PyTorchHandle).module
        val tensor = Tensor.fromBlob(data, shape)
        val output = module.forward(IValue.from(tensor)).toTensor()
        return output.dataAsFloatArray
    }
}
