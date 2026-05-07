# Edge-model build flow (TANUH interim)

This directory holds the offline encryption tool and per-build artefacts for the on-device
edge-model integration in the `tanuh` Gradle flavour. It exists so that a TANUH developer can
produce a signed APK with their proprietary model bundled and AES-GCM-encrypted, on their own
machine, in two commands.

The on-device runtime is **PyTorch Mobile 1.13.1** (`org.pytorch:pytorch_android:1.13.1`).
Versions are pinned exactly to the AI team's PoC at `~/IdeaProjects/aiapp` because clinical
accuracy was validated against that exact runtime; converting to a different runtime (e.g.
TFLite) would force re-validation.

The full design is documented at `~/.claude/plans/composed-tumbling-bachman.md`.

## Threat model — read first

This is the **interim** build. The encrypted model **and the AES key both ship in the same
APK** (the key lives in `registry.json` inside `assets/models/`). That is *obfuscation*,
not the §5.1 protection in `~/.claude/plans/frolicking-pondering-marble.md`:

- It defeats casual extraction (`unzip the APK, grab the .pt` no longer works).
- It does *not* defeat a determined reverser who reads the bundled key and decrypts.

There is also a brief **on-disk plaintext window** at load time: PyTorch Mobile's
`Module.load(path)` requires a file path, not a buffer, so the encrypted blob is decrypted
into the app's private `filesDir/<modelKey>.pt` (mode 0600), loaded, and the file is deleted
in a `finally` block. Plaintext exists on disk for the duration of `Module.load` only. This
is consistent with the obfuscation posture above; a determined reverser has easier paths.

The proper defence — encrypted blob in TANUH's S3, key in `organisation_config`, plus a
custom JNI shim that calls `torch::jit::load(istream)` to keep plaintext entirely off-disk —
arrives in a later iteration. Use this build for trainings, demos, and pre-go-live
validation, not for unrestricted public distribution.

## One-time setup

Requires Node 20 (already pinned via `.nvmrc`), Java 17, Android SDK build-tools 35.0.0
and NDK 27.1.12297006 (matches the rest of the repo).

```bash
source ~/.nvm/nvm.sh && nvm use 20

# Install JS dependencies, apply patches, run prebuild. Required on a fresh checkout;
# repeat only when package.json or patches change.
make deps

# Generate a release keystore for signing the tanuh APK (keystore stays local; never committed).
make tanuh-setup
```

`make tanuh-setup` creates `tanuh-release-key.keystore` in the repo root and prompts for
the keystore + key passwords. Export them as env vars before each build:

```bash
export tanuh_KEYSTORE_PASSWORD='…'
export tanuh_KEY_ALIAS='tanuh'
```

## Per-build flow

1. **Drop the plaintext model in the source dir.** Anything under
   `tools/edge-model/source/*.pt` is gitignored.

   ```bash
   cp /path/to/mvit2_fold5_2_latest_traced.pt tools/edge-model/source/
   ```

2. **Build the signed APK.**

   ```bash
   make tanuh-apk
   ```

   This target chains:
   - `tanuh-encrypt`: runs `node tools/edge-model/encrypt-model.js`, encrypts the source
     model with a fresh AES-GCM-256 key, writes the encrypted blob and `registry.json`
     into `packages/openchs-android/android/app/src/tanuh/assets/models/` (gitignored).
   - `assembleTanuhRelease`: Gradle release build for the `tanuh` flavour, signed with
     the keystore from `make tanuh-setup`.

   The signed APK lands at:

   ```
   packages/openchs-android/android/app/build/outputs/apk/tanuh/release/app-tanuh-release.apk
   ```

3. **Distribute.** Upload the APK to gdrive (or wherever the TANUH programme team
   distributes from). The plaintext model never leaves the build machine.

## Quick-iteration debug build (no encryption, no signing)

For JS / native iteration with the real TANUH model:

```bash
make tanuh-encrypt TANUH_MODEL_KEY=mvit2_fold5_2_latest_traced  # populate src/tanuh/assets/models/
make run_packager                                               # in another terminal
make run_app_tanuh_dev                                          # debug build, prod backend, dev menu
```

Skip the `tanuh-encrypt` step if you don't need the model loaded — the app will still boot
but `EdgeModelService` will refuse inference for any unknown `modelKey`.

### Placeholder model for local development (no Python, no TANUH .pt)

If you don't have access to TANUH's `.pt` yet but want to exercise the full pipeline:

```bash
make tanuh-placeholder      # downloads a 20 MB public MobileNetV2 .pt, encrypts as 'placeholder'
make run_packager
make run_app_tanuh_dev
```

`make tanuh-placeholder` curls PyTorch's official `HelloWorldApp/model.pt` (ImageNet
224×224 RGB, 1000-class) and registers it under `placeholder` using
`tools/edge-model/placeholder-override.json` (`imagenet-rgb-chw` + `argmax-labels`).
Inference values are unrelated to TANUH's domain — use this to validate that the bridge,
preprocessor, decoder, encryption, and on-disk-decrypt-then-load flow all work end-to-end.

In a decision rule:

```js
const result = await params.services.edgeModelService.runInferenceOnImage(
    'placeholder', imagePath
);
// result: { label: <classIndex>, confidence, classIndex, raw: number[1000] }
```

Alternative — if you'd rather generate a tiny stub model that matches the real MViT2 shape
exactly (`[1, 3, 256, 256]` BGR/CHW → single logit), see
`tools/edge-model/make-placeholder-pt.py` (requires PyTorch installed locally).

## What lives where

```
tools/edge-model/
├─ encrypt-model.js              # AES-GCM-256 encryption CLI — format-agnostic
├─ sample-override.json          # ImageNet-style sample (imagenet-rgb-chw + argmax-labels)
├─ tanuh-mvit2-override.json     # PoC pipeline for TANUH's MViT2 model — used by `make tanuh-apk` by default
├─ source/                       # plaintext .pt / .tflite files — gitignored
└─ README.md                     # this file

packages/openchs-android/android/app/src/tanuh/
├─ assets/models/                # encrypted blob + registry.json — gitignored, build-time only
├─ res/                          # branding (icons / splash) — replace generic placeholders with TANUH assets
└─ README.md

packages/openchs-android/android/app/src/main/java/com/openchsclient/
├─ EdgeModelModule.kt            # React Native bridge — engine-agnostic, model-agnostic
├─ ModelContract.kt              # parses the override JSON DSL
├─ engine/                       # InferenceEngine interface + PyTorchEngine
├─ preprocessing/                # ImagePreprocessor interface + named registry
└─ decoding/                     # OutputDecoder interface + named registry
```

## When the model changes

Re-run `make tanuh-apk`. The encryption CLI generates a fresh AES key and IV per run,
so old encrypted blobs become invalid. This is intentional — there's no key-rotation
ambiguity, the build is reproducible from the plaintext source.

## The plugin model — adding a new model with novel preprocessing

The bridge is **engine-agnostic** and **model-agnostic**. Per-model semantics — which
inference engine to use, how to preprocess the image, how to decode the output — are
declared as a small JSON DSL in the registry's `override` block:

```json
{
  "engine": "pytorch",
  "input":  { "preprocessor": "<name>", "params": { … } },
  "output": { "decoder":      "<name>", "params": { … } }
}
```

Plugin names resolve to Kotlin classes in:
- `preprocessing/Preprocessors.kt` — `imagenet-rgb-chw`, `mean-target-bgr-rounded`
- `decoding/Decoders.kt`         — `argmax-labels`, `sigmoid-binary`, `raw-floats`

To add a new model:

1. **Reuse an existing plugin if possible.** Most preprocessing variants can be expressed
   as different `params` for an existing plugin (different size, mean/std, channel order).
   Write a new override JSON pointing at the same plugin name with the new params.

2. **If genuinely novel:** drop a new class implementing `ImagePreprocessor` (or
   `OutputDecoder`) into the relevant registry, register it by name in the `REGISTRY` map,
   and reference the new name from your override JSON. **No edits to `EdgeModelModule.kt`.**

The DSL is **pure data** — no executable code. It travels in the AES-encrypted bundle and
parses through `org.json.JSONObject` only; there is no `eval`, no JS callback, no
dynamic class loading.

## Adding a new inference engine (e.g. ExecuTorch, ONNX Runtime, TFLite)

1. Add the engine's Gradle dependency to `app/build.gradle`.
2. Implement `InferenceEngine` in a new class under `engine/`.
3. Register it in `EdgeModelModule.engines` (the `engine` field in JSON now matches the new key).

`EdgeModelModule.kt` itself does not need to change — only the engine inventory map at
construction time.
