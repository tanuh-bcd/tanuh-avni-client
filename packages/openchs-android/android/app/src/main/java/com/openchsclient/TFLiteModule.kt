package com.openchsclient

import android.content.ComponentCallbacks2
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import com.facebook.react.bridge.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * Generic TFLite inference bridge for the Avni client.
 *
 * Overall design (~/.claude/plans/composed-tumbling-bachman.md):
 *   • The native module is **model-agnostic** — preprocessing parameters come from the
 *     resolved `ModelContract`, not hardcoded constants. A model is identified by an
 *     opaque `modelKey`; the JS side consults `assets/models/registry.json` to map
 *     keys to asset specs (plain or AES-GCM-encrypted).
 *   • Models load **lazily on first inference** for a given key. Once loaded the
 *     interpreter stays for the app's lifetime, *until* the OS asks for memory back via
 *     `onTrimMemory(TRIM_MEMORY_RUNNING_LOW)` or worse, at which point we close all
 *     interpreters and free the off-heap buffers. Subsequent inferences self-heal: the
 *     load-args cache lets us rebuild without round-tripping to JS.
 *   • Encrypted models are **never written to disk in plaintext**. The encrypted blob
 *     is memory-mapped from the APK, stream-decrypted via chunked `Cipher.update` into
 *     a direct off-heap `ByteBuffer`, and handed to the `Interpreter`. Java `byte[]`
 *     scratch space is zeroed after copy. AES-GCM authenticates ciphertext; we additionally
 *     verify SHA-256 of the plaintext against the value the encryption CLI recorded, so a
 *     swapped blob fails fast.
 *   • The module registers `ComponentCallbacks2` to receive memory-pressure signals.
 *     Backgrounding the app for a camera intent or phone call does *not* trigger eviction
 *     unless the OS reports actual pressure — the common case (form → camera → return →
 *     run inference) keeps the interpreter warm.
 *
 * JS-facing API (via NativeModules.TFLiteModule):
 *   getRegistry(): Promise<object>
 *   loadModel(modelKey, assetPath, overrideJson|null): Promise<boolean>
 *   loadEncryptedModel(modelKey, encryptedAssetPath, base64Key, sha256, overrideJson|null): Promise<boolean>
 *   runInference(modelKey, inputData: number[]): Promise<number[]>
 *   runInferenceOnImage(modelKey, imagePath): Promise<number[]>
 */
class TFLiteModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext), ComponentCallbacks2 {

    companion object {
        private const val TAG = "TFLiteModule"
        private const val REGISTRY_ASSET = "models/registry.json"
        private const val GCM_TAG_BITS = 128                  // AES-GCM authentication-tag size
        private const val GCM_IV_BYTES = 12                   // 96-bit IV — recommended for GCM
        private const val DECRYPT_CHUNK_BYTES = 64 * 1024     // 64 KB chunks: balance between syscall overhead and Java-heap pressure
    }

    /** Live interpreter cache. Cleared on memory-pressure eviction. */
    private val interpreters = HashMap<String, Interpreter>()

    /** Resolved per-model contract (preprocessing semantics + labels). Cleared on eviction. */
    private val contracts = HashMap<String, ModelContract>()

    /**
     * Cached load arguments — survives memory-pressure eviction so we can self-heal a
     * subsequent inference call without bouncing back to JS for the encryption key etc.
     * The AES key persists in native memory between eviction and reload; this matches the
     * existing security posture (the key was already in native memory while the model was loaded).
     */
    private val loadArgs = HashMap<String, LoadArgs>()

    private sealed class LoadArgs {
        abstract val overrideJson: String?
        data class Plain(val assetPath: String, override val overrideJson: String?) : LoadArgs()
        data class Encrypted(
            val encryptedAssetPath: String,
            val base64Key: String,
            val sha256Hex: String,
            override val overrideJson: String?
        ) : LoadArgs()
    }

    init {
        // Subscribe to OS memory-pressure callbacks. Application context (not activity) so we
        // get signals even while the activity is paused (e.g. behind the camera intent).
        reactContext.applicationContext.registerComponentCallbacks(this)
    }

    override fun getName(): String = "TFLiteModule"

    // ── React-callable methods ─────────────────────────────────────────────────────────

    /**
     * Read the per-flavour model registry (`assets/models/registry.json`) and return its
     * parsed contents. Called once at app boot from `EdgeModelService.init()`. The JS side
     * caches the result and uses it to resolve `modelKey` → asset spec on each inference.
     */
    @ReactMethod
    fun getRegistry(promise: Promise) {
        try {
            val raw = reactApplicationContext.assets.open(REGISTRY_ASSET).bufferedReader().use { it.readText() }
            val map = jsonStringToWritableMap(raw)
            promise.resolve(map)
        } catch (e: Exception) {
            Log.e(TAG, "getRegistry: failed to read $REGISTRY_ASSET: ${e.message}", e)
            promise.reject("REGISTRY_LOAD_ERROR", "Failed to read $REGISTRY_ASSET: ${e.message}", e)
        }
    }

    /**
     * Load a plaintext .tflite from APK assets. Idempotent — calling twice for the same
     * `modelKey` is a no-op. The `overrideJson` argument is consulted only when the model
     * carries no embedded TFLite metadata.
     */
    @ReactMethod
    fun loadModel(modelKey: String, assetPath: String, overrideJson: String?, promise: Promise) {
        try {
            loadArgs[modelKey] = LoadArgs.Plain(assetPath, overrideJson)
            ensureLoaded(modelKey)
            promise.resolve(true)
        } catch (e: Exception) {
            Log.e(TAG, "loadModel($modelKey): ${e.message}", e)
            promise.reject("TFLITE_LOAD_ERROR", "Failed to load model '$modelKey': ${e.message}", e)
        }
    }

    /**
     * Load an AES-GCM-encrypted .tflite from APK assets. Plaintext is held only in a direct
     * off-heap `ByteBuffer` for the interpreter's lifetime; never written to disk.
     */
    @ReactMethod
    fun loadEncryptedModel(
        modelKey: String,
        encryptedAssetPath: String,
        base64Key: String,
        sha256Hex: String,
        overrideJson: String?,
        promise: Promise
    ) {
        try {
            loadArgs[modelKey] = LoadArgs.Encrypted(encryptedAssetPath, base64Key, sha256Hex, overrideJson)
            ensureLoaded(modelKey)
            promise.resolve(true)
        } catch (e: Exception) {
            Log.e(TAG, "loadEncryptedModel($modelKey): ${e.message}", e)
            promise.reject("TFLITE_LOAD_ERROR", "Failed to load encrypted model '$modelKey': ${e.message}", e)
        }
    }

    /**
     * Run inference on a pre-processed float array. Caller is responsible for shape
     * conformance — we validate against the cached interpreter's input tensor.
     */
    @ReactMethod
    fun runInference(modelKey: String, inputData: ReadableArray, promise: Promise) {
        try {
            ensureLoaded(modelKey)
            val interpreter = interpreters[modelKey]!!
            val inputBuffer = buildInputBufferFromArray(interpreter, inputData)
            val (outputBuffer, outputSize) = buildOutputBuffer(interpreter)
            interpreter.run(inputBuffer, outputBuffer)
            promise.resolve(readOutputBuffer(modelKey, outputBuffer, outputSize))
        } catch (e: Exception) {
            Log.e(TAG, "runInference($modelKey): ${e.message}", e)
            promise.reject("TFLITE_INFERENCE_ERROR", "Inference failed: ${e.message}", e)
        }
    }

    /**
     * Run inference directly on an image file. Image preprocessing (decode → resize →
     * normalise → layout transpose) is driven entirely by the cached `ModelContract`;
     * no per-model code lives here.
     */
    @ReactMethod
    fun runInferenceOnImage(modelKey: String, imagePath: String, promise: Promise) {
        try {
            ensureLoaded(modelKey)
            val interpreter = interpreters[modelKey]!!
            val contract = contracts[modelKey]!!.input

            val raw = BitmapFactory.decodeFile(imagePath)
                ?: throw IllegalArgumentException("Cannot decode image at '$imagePath'. Check the path and file format.")

            val resized = if (raw.width == contract.width && raw.height == contract.height) raw
                else Bitmap.createScaledBitmap(raw, contract.width, contract.height, true)

            val inputBuffer = preprocessImageToBuffer(resized, contract)
            val (outputBuffer, outputSize) = buildOutputBuffer(interpreter)
            interpreter.run(inputBuffer, outputBuffer)
            promise.resolve(readOutputBuffer(modelKey, outputBuffer, outputSize))
        } catch (e: Exception) {
            Log.e(TAG, "runInferenceOnImage($modelKey): ${e.message}", e)
            promise.reject("TFLITE_INFERENCE_ERROR", "Image inference failed: ${e.message}", e)
        }
    }

    // ── Lifecycle hooks ────────────────────────────────────────────────────────────────

    /**
     * Memory-pressure callback. Closes all interpreters and clears their off-heap buffers
     * when the OS asks for memory back. We deliberately do *not* clear `loadArgs`, so the
     * next inference can self-heal-reload without involving JS.
     */
    override fun onTrimMemory(level: Int) {
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW) {
            Log.w(TAG, "onTrimMemory(level=$level) — releasing ${interpreters.size} interpreter(s); load-args retained for self-heal reload")
            interpreters.values.forEach { it.close() }
            interpreters.clear()
            contracts.clear()
        }
    }

    override fun onLowMemory() {
        // Older API — delegate to the same eviction path.
        onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
    }

    override fun onConfigurationChanged(newConfig: Configuration) { /* no-op */ }

    override fun onCatalystInstanceDestroy() {
        Log.d(TAG, "onCatalystInstanceDestroy: closing ${interpreters.size} interpreter(s)")
        interpreters.values.forEach { it.close() }
        interpreters.clear()
        contracts.clear()
        loadArgs.clear()
        reactApplicationContext.applicationContext.unregisterComponentCallbacks(this)
    }

    // ── Private helpers ───────────────────────────────────────────────────────────────

    /**
     * Build the interpreter for `modelKey` if it isn't already loaded. Self-healing:
     * after a memory-pressure eviction, the next inference call hits this path and rebuilds
     * the interpreter from the cached `loadArgs` without re-prompting JS.
     */
    private fun ensureLoaded(modelKey: String) {
        if (interpreters.containsKey(modelKey)) return
        val args = loadArgs[modelKey]
            ?: throw IllegalStateException(
                "Model not loaded: '$modelKey'. Call loadModel()/loadEncryptedModel() before inference."
            )
        when (args) {
            is LoadArgs.Plain -> buildInterpreter(modelKey, mmapAsset(args.assetPath), args.overrideJson)
            is LoadArgs.Encrypted -> {
                val plaintext = decryptToDirectBuffer(args.encryptedAssetPath, args.base64Key, args.sha256Hex)
                buildInterpreter(modelKey, plaintext, args.overrideJson)
            }
        }
    }

    private fun buildInterpreter(modelKey: String, modelBuffer: ByteBuffer, overrideJson: String?) {
        Log.d(TAG, "buildInterpreter($modelKey): size=${modelBuffer.capacity()} bytes")
        val options = Interpreter.Options().apply { numThreads = 2 }
        val interpreter = Interpreter(modelBuffer, options)
        val contract = ModelContract.resolve(modelBuffer, interpreter, overrideJson)

        val it = interpreter.getInputTensor(0)
        val ot = interpreter.getOutputTensor(0)
        Log.d(TAG, "buildInterpreter($modelKey): input=${it.shape().toList()} ${it.dataType()}, output=${ot.shape().toList()} ${ot.dataType()}")
        Log.d(TAG, "buildInterpreter($modelKey): contract.input=${contract.input}, contract.output.labels=${contract.output.labels}")

        interpreters[modelKey] = interpreter
        contracts[modelKey] = contract
    }

    /** Memory-map an asset file as read-only. Lazy paging; no Java-heap allocation. */
    private fun mmapAsset(assetPath: String): MappedByteBuffer {
        val fd = reactApplicationContext.assets.openFd(assetPath)
        FileInputStream(fd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    /**
     * Stream-decrypt an AES-GCM-encrypted asset into a direct off-heap `ByteBuffer`.
     *
     * Layout of the on-disk blob (matches `tools/edge-model/encrypt-model.js`):
     *   [12-byte IV][ciphertext...][16-byte GCM tag]
     *
     * We decrypt in `DECRYPT_CHUNK_BYTES` chunks via `Cipher.update`, writing directly into
     * a preallocated direct buffer to avoid a single ~100 MB Java-heap allocation. After
     * `doFinal`, integrity is verified via SHA-256 of the resulting plaintext against the
     * value the encryption CLI recorded — defends against blob substitution as well as
     * silent corruption.
     */
    private fun decryptToDirectBuffer(encryptedAssetPath: String, base64Key: String, expectedSha256Hex: String): ByteBuffer {
        val fd = reactApplicationContext.assets.openFd(encryptedAssetPath)
        val totalLen = fd.declaredLength
        val ciphertextLen = totalLen - GCM_IV_BYTES
        val plaintextLen = (ciphertextLen - GCM_TAG_BITS / 8).toInt()

        Log.d(TAG, "decryptToDirectBuffer($encryptedAssetPath): total=$totalLen, plaintext=$plaintextLen bytes")

        val mapped = FileInputStream(fd.fileDescriptor).use { fis ->
            fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, totalLen)
        }
        // First 12 bytes = IV
        val iv = ByteArray(GCM_IV_BYTES).also { mapped.get(it) }

        val key = Base64.decode(base64Key, Base64.DEFAULT)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(GCM_TAG_BITS, iv))

        val output = ByteBuffer.allocateDirect(plaintextLen).order(ByteOrder.nativeOrder())
        val chunk = ByteArray(DECRYPT_CHUNK_BYTES)
        val outChunk = ByteArray(DECRYPT_CHUNK_BYTES + 32)
        var remaining = ciphertextLen.toInt()
        while (remaining > 0) {
            val n = minOf(remaining, DECRYPT_CHUNK_BYTES)
            mapped.get(chunk, 0, n)
            val produced = cipher.update(chunk, 0, n, outChunk)
            if (produced > 0) output.put(outChunk, 0, produced)
            remaining -= n
        }
        // doFinal verifies the GCM tag — throws AEADBadTagException on tamper or wrong key.
        val tail = cipher.doFinal()
        if (tail.isNotEmpty()) output.put(tail)
        output.rewind()

        // Zero the temporary Java byte arrays — we cannot un-page the mmap'd ciphertext or
        // the direct buffer (TFLite owns it now), but we can clean up our scratch space.
        chunk.fill(0); outChunk.fill(0); key.fill(0)

        // Plaintext integrity check — defends against a swapped blob with valid GCM auth
        // (e.g. someone re-encrypted a different model with the same key).
        val md = MessageDigest.getInstance("SHA-256")
        // We have to re-iterate the buffer since MessageDigest doesn't accept a ByteBuffer slice cheaply.
        val view = output.duplicate()
        val readBuf = ByteArray(DECRYPT_CHUNK_BYTES)
        while (view.hasRemaining()) {
            val n = minOf(view.remaining(), readBuf.size)
            view.get(readBuf, 0, n)
            md.update(readBuf, 0, n)
        }
        val actual = md.digest().joinToString("") { "%02x".format(it) }
        readBuf.fill(0)
        if (!actual.equals(expectedSha256Hex, ignoreCase = true)) {
            throw SecurityException("Decrypted plaintext SHA-256 mismatch (expected=$expectedSha256Hex, actual=$actual)")
        }

        return output
    }

    private fun buildInputBufferFromArray(interpreter: Interpreter, inputData: ReadableArray): ByteBuffer {
        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        val inputSize = inputShape.fold(1) { acc, dim -> acc * dim }
        if (inputData.size() != inputSize) {
            throw IllegalArgumentException(
                "Input size mismatch: model expects $inputSize floats ${inputShape.toList()} but received ${inputData.size()}."
            )
        }
        val buffer = ByteBuffer.allocateDirect(inputSize * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
        for (i in 0 until inputData.size()) buffer.putFloat(inputData.getDouble(i).toFloat())
        buffer.rewind()
        return buffer
    }

    /**
     * Bitmap → direct float32 ByteBuffer, driven entirely by the resolved contract.
     *
     * For each pixel: extract R/G/B, divide by 255 (or apply contract.scale), subtract mean,
     * divide by std. Channel-first vs channel-last layout writes the same data in a different
     * stride pattern.
     */
    private fun preprocessImageToBuffer(bitmap: Bitmap, input: ModelContract.InputSpec): ByteBuffer {
        val w = input.width
        val h = input.height
        val c = input.channels
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val buffer = ByteBuffer.allocateDirect(c * h * w * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())

        if (input.layout == ModelContract.Layout.CHW) {
            for (channel in 0 until c) {
                for (row in 0 until h) {
                    for (col in 0 until w) {
                        buffer.putFloat(normalizedChannelValue(pixels[row * w + col], channel, input))
                    }
                }
            }
        } else { // HWC
            for (row in 0 until h) {
                for (col in 0 until w) {
                    val pixel = pixels[row * w + col]
                    for (channel in 0 until c) {
                        buffer.putFloat(normalizedChannelValue(pixel, channel, input))
                    }
                }
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun normalizedChannelValue(pixel: Int, channel: Int, input: ModelContract.InputSpec): Float {
        val raw = when (channel) {
            0 -> (pixel shr 16) and 0xFF  // R
            1 -> (pixel shr 8) and 0xFF   // G
            2 -> pixel and 0xFF           // B
            else -> 0
        }
        val scaled = raw * input.scale
        val m = if (channel < input.mean.size) input.mean[channel] else 0f
        val s = if (channel < input.std.size && input.std[channel] != 0f) input.std[channel] else 1f
        return (scaled - m) / s
    }

    private fun buildOutputBuffer(interpreter: Interpreter): Pair<ByteBuffer, Int> {
        val outputTensor = interpreter.getOutputTensor(0)
        val outputSize = outputTensor.shape().fold(1) { acc, dim -> acc * dim }
        val buffer = ByteBuffer.allocateDirect(outputSize * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
        return Pair(buffer, outputSize)
    }

    /**
     * Read the output tensor as a flat float array and forward to JS as a number[]. If the
     * contract carries class labels and the output is a 1-D vector with matching length,
     * log the predicted class for debugging — purely diagnostic, never affects the return value.
     */
    private fun readOutputBuffer(modelKey: String, outputBuffer: ByteBuffer, outputSize: Int): WritableArray {
        outputBuffer.rewind()
        val results = WritableNativeArray()
        val values = FloatArray(outputSize)
        for (i in 0 until outputSize) {
            values[i] = outputBuffer.float
            results.pushDouble(values[i].toDouble())
        }
        val labels = contracts[modelKey]?.output?.labels
        if (!labels.isNullOrEmpty() && labels.size == outputSize) {
            val argmax = values.indices.maxByOrNull { values[it] } ?: 0
            Log.d(TAG, "readOutputBuffer($modelKey): predicted=${labels[argmax]} score=${values[argmax]}")
        }
        return results
    }

    private fun jsonStringToWritableMap(raw: String): WritableMap {
        val obj = org.json.JSONObject(raw)
        return jsonObjectToWritableMap(obj)
    }

    private fun jsonObjectToWritableMap(obj: org.json.JSONObject): WritableMap {
        val map = Arguments.createMap()
        val keys = obj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            when (val v = obj.opt(key)) {
                null, org.json.JSONObject.NULL -> map.putNull(key)
                is org.json.JSONObject -> map.putMap(key, jsonObjectToWritableMap(v))
                is org.json.JSONArray -> map.putArray(key, jsonArrayToWritableArray(v))
                is Boolean -> map.putBoolean(key, v)
                is Int -> map.putInt(key, v)
                is Long -> map.putDouble(key, v.toDouble())
                is Double -> map.putDouble(key, v)
                is String -> map.putString(key, v)
                else -> map.putString(key, v.toString())
            }
        }
        return map
    }

    private fun jsonArrayToWritableArray(arr: org.json.JSONArray): WritableArray {
        val out = Arguments.createArray()
        for (i in 0 until arr.length()) {
            when (val v = arr.opt(i)) {
                null, org.json.JSONObject.NULL -> out.pushNull()
                is org.json.JSONObject -> out.pushMap(jsonObjectToWritableMap(v))
                is org.json.JSONArray -> out.pushArray(jsonArrayToWritableArray(v))
                is Boolean -> out.pushBoolean(v)
                is Int -> out.pushInt(v)
                is Long -> out.pushDouble(v.toDouble())
                is Double -> out.pushDouble(v)
                is String -> out.pushString(v)
                else -> out.pushString(v.toString())
            }
        }
        return out
    }
}
