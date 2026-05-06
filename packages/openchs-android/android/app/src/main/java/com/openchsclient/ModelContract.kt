package com.openchsclient

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.nio.ByteBuffer

/**
 * Resolved input/output contract for a TFLite model.
 *
 * Why this class exists
 * ─────────────────────
 * The TFLite `Interpreter` knows the *tensor shape* and *dtype* of inputs and outputs, but it
 * does not know the *preprocessing semantics* — image normalisation (mean / std / scale),
 * image layout (CHW vs HWC), or human-readable output labels. A `Bitmap` cannot be fed to
 * the interpreter without these.
 *
 * Two sources are supported, in priority order. This is Option 2 from the plan
 * (~/.claude/plans/composed-tumbling-bachman.md):
 *
 *   1. **Embedded TFLite Model Metadata** (FlatBuffer inside the .tflite, as defined by
 *      Google's `metadata_schema.fbs`). Models exported by TFLite Model Maker, MediaPipe,
 *      and many vendors carry this. Read via `MetadataExtractor`.
 *
 *   2. **Per-model `override` block** in `registry.json`. Used for vendor-supplied models
 *      that don't carry embedded metadata. Lets us bridge models that pre-date or skip
 *      Google's metadata convention.
 *
 * The current placeholder (`edge_model.tflite`) has no embedded metadata, so it uses (2)
 * with the override block defined in `tools/edge-model/sample-override.json`. TANUH's
 * eventual production model is expected to ship with (1).
 */
data class ModelContract(
    val input: InputSpec,
    val output: OutputSpec
) {
    data class InputSpec(
        val type: String,        // "image" | "raw" — drives whether image preprocessing applies
        val width: Int,          // image only (0 for raw)
        val height: Int,         // image only
        val channels: Int,       // image only — typically 3 (RGB)
        val layout: Layout,      // image only — pixel ordering in the input tensor
        val dtype: DType,
        val scale: Float,        // raw_pixel * scale before mean/std (typically 1/255 = 0.003921…)
        val mean: FloatArray,    // length = channels; ImageNet defaults [0.485, 0.456, 0.406]
        val std: FloatArray      // length = channels; ImageNet defaults [0.229, 0.224, 0.225]
    )

    data class OutputSpec(
        val shape: IntArray,     // tensor shape from interpreter (always trustworthy)
        val dtype: DType,
        val labels: List<String>?  // class names if available; null otherwise
    )

    enum class Layout {
        /** (1, channels, height, width) — PyTorch / channels-first */
        CHW,
        /** (1, height, width, channels) — TFLite / Keras default / channels-last */
        HWC
    }

    enum class DType { FLOAT32 } // only float32 supported in this iteration

    companion object {
        private const val TAG = "ModelContract"

        /**
         * Resolve the contract for a freshly-built interpreter.
         *
         * Priority: embedded metadata > override JSON > error. Failing loudly is intentional —
         * a model with neither cannot be safely preprocessed.
         */
        fun resolve(modelBuffer: ByteBuffer, interpreter: Interpreter, overrideJson: String?): ModelContract {
            val fromMetadata = tryFromMetadata(modelBuffer, interpreter)
            if (fromMetadata != null) {
                Log.d(TAG, "resolve: using embedded TFLite metadata")
                return fromMetadata
            }
            if (overrideJson != null) {
                Log.d(TAG, "resolve: no embedded metadata; using 'override' block from registry.json")
                return fromOverride(overrideJson, interpreter)
            }
            throw IllegalStateException(
                "Model has no embedded TFLite metadata and registry.json provides no 'override' block. " +
                "Either embed metadata via TFLite Model Maker / MediaPipe, or add an 'override' entry " +
                "in registry.json describing input preprocessing and output labels."
            )
        }

        /**
         * Best-effort read of embedded TFLite Model Metadata. Returns null on any failure so
         * the caller can fall back to the override path. This is intentionally defensive —
         * the FlatBuffer accessor surface differs across model versions and a parse failure
         * here should not crash inference if a usable override exists.
         */
        private fun tryFromMetadata(modelBuffer: ByteBuffer, interpreter: Interpreter): ModelContract? {
            return try {
                val extractor = MetadataExtractor(modelBuffer)
                if (!extractor.hasMetadata()) {
                    Log.d(TAG, "tryFromMetadata: model carries no embedded metadata")
                    return null
                }

                // Tensor shape and dtype are always pulled from the interpreter (authoritative);
                // metadata only contributes the preprocessing parameters and labels.
                val inputTensor = interpreter.getInputTensor(0)
                val outputTensor = interpreter.getOutputTensor(0)
                val shape = inputTensor.shape()

                val (height, width, channels, layout) = inferImageGeometry(shape)
                val (scale, mean, std) = readNormalization(extractor, channels)
                val labels = readLabels(extractor)

                ModelContract(
                    input = InputSpec(
                        type = if (shape.size == 4) "image" else "raw",
                        width = width, height = height, channels = channels,
                        layout = layout,
                        dtype = DType.FLOAT32,
                        scale = scale, mean = mean, std = std
                    ),
                    output = OutputSpec(
                        shape = outputTensor.shape(),
                        dtype = DType.FLOAT32,
                        labels = labels
                    )
                )
            } catch (e: Exception) {
                Log.w(TAG, "tryFromMetadata: extractor or parse failed (${e.message}); falling back to override if available")
                null
            }
        }

        private data class ImageGeometry(val height: Int, val width: Int, val channels: Int, val layout: Layout)

        /**
         * Disambiguate CHW vs HWC from the tensor shape. RGB images have either
         * shape (1, 3, H, W) [CHW] or (1, H, W, 3) [HWC]; the channel dim is the small one.
         */
        private fun inferImageGeometry(shape: IntArray): ImageGeometry {
            if (shape.size != 4) return ImageGeometry(0, 0, 0, Layout.HWC)
            val channelsFirst = shape[1] in setOf(1, 3, 4) && shape[3] !in setOf(1, 3, 4)
            return if (channelsFirst) {
                ImageGeometry(height = shape[2], width = shape[3], channels = shape[1], layout = Layout.CHW)
            } else {
                ImageGeometry(height = shape[1], width = shape[2], channels = shape[3], layout = Layout.HWC)
            }
        }

        /**
         * Read mean / std from `NormalizationOptions` if present in metadata. The TFLite
         * metadata schema places these inside a `processUnits` collection on the input
         * tensor metadata. We use reflection-friendly access here because the FlatBuffer
         * accessor signatures vary across schema versions.
         */
        private fun readNormalization(extractor: MetadataExtractor, channels: Int): Triple<Float, FloatArray, FloatArray> {
            return try {
                val inputMeta = extractor.getInputTensorMetadata(0) ?: return defaultNormalization(channels)
                val units = inputMeta.javaClass.getMethod("processUnitsLength").invoke(inputMeta) as Int
                for (i in 0 until units) {
                    val unit = inputMeta.javaClass.getMethod("processUnits", Int::class.javaPrimitiveType).invoke(inputMeta, i)
                    val opts = unit?.javaClass?.getMethod("options")?.invoke(unit) ?: continue
                    val meanArr = opts.javaClass.getMethod("meanLength").invoke(opts) as Int
                    val stdArr  = opts.javaClass.getMethod("stdLength").invoke(opts)  as Int
                    if (meanArr > 0 && stdArr > 0) {
                        val mean = FloatArray(meanArr) { idx ->
                            opts.javaClass.getMethod("mean", Int::class.javaPrimitiveType).invoke(opts, idx) as Float
                        }
                        val std = FloatArray(stdArr) { idx ->
                            opts.javaClass.getMethod("std", Int::class.javaPrimitiveType).invoke(opts, idx) as Float
                        }
                        return Triple(1f, mean, std)
                    }
                }
                defaultNormalization(channels)
            } catch (e: Exception) {
                Log.w(TAG, "readNormalization: ${e.message}; using identity normalization")
                defaultNormalization(channels)
            }
        }

        private fun defaultNormalization(channels: Int): Triple<Float, FloatArray, FloatArray> =
            Triple(1f, FloatArray(channels.coerceAtLeast(1)), FloatArray(channels.coerceAtLeast(1)) { 1f })

        /** Read the first associated `.txt` file as one label per line. */
        private fun readLabels(extractor: MetadataExtractor): List<String>? {
            return try {
                val files = extractor.associatedFileNames ?: return null
                val labelFile = files.firstOrNull { it.endsWith(".txt", ignoreCase = true) } ?: return null
                extractor.getAssociatedFile(labelFile).bufferedReader().useLines { it.toList() }
            } catch (e: Exception) {
                Log.w(TAG, "readLabels: failed (${e.message})")
                null
            }
        }

        /**
         * Build the contract from the `override` block in registry.json.
         *
         * Used for vendor models without embedded metadata (the current placeholder is one
         * such model). The override carries everything the metadata path would have given
         * us: image size, channel count, layout, normalisation, output labels.
         */
        private fun fromOverride(overrideJson: String, interpreter: Interpreter): ModelContract {
            val root = JSONObject(overrideJson)
            val inputJson = root.getJSONObject("input")
            val outputJson = root.getJSONObject("output")

            val norm = inputJson.optJSONObject("normalization")
            val channels = inputJson.optInt("channels", 3)
            val mean = norm?.optJSONArray("mean")?.toFloatArray() ?: FloatArray(channels)
            val std = norm?.optJSONArray("std")?.toFloatArray() ?: FloatArray(channels) { 1f }
            val scale = norm?.optDouble("scale", 1.0)?.toFloat() ?: 1f

            val outputTensor = interpreter.getOutputTensor(0)

            return ModelContract(
                input = InputSpec(
                    type = inputJson.optString("type", "raw"),
                    width = inputJson.optInt("width", 0),
                    height = inputJson.optInt("height", 0),
                    channels = channels,
                    layout = if (inputJson.optString("layout", "HWC") == "CHW") Layout.CHW else Layout.HWC,
                    dtype = DType.FLOAT32,
                    scale = scale,
                    mean = mean,
                    std = std
                ),
                output = OutputSpec(
                    shape = outputTensor.shape(),
                    dtype = DType.FLOAT32,
                    labels = outputJson.optJSONArray("labels")?.toStringList()
                )
            )
        }

        private fun JSONArray.toFloatArray(): FloatArray =
            FloatArray(length()) { i -> getDouble(i).toFloat() }

        private fun JSONArray.toStringList(): List<String> =
            List(length()) { i -> getString(i) }
    }
}
