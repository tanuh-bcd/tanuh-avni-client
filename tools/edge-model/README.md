# Edge-model build flow (TANUH interim)

This directory holds the offline encryption tool and per-build artefacts for the on-device
TFLite edge-model integration in the `tanuh` Gradle flavour. It exists so that a TANUH
developer can produce a signed APK with their proprietary model bundled and AES-GCM-encrypted,
on their own machine, in two commands.

The full design is documented at `~/.claude/plans/composed-tumbling-bachman.md`.

## Threat model — read first

This is the **interim** build. The encrypted model **and the AES key both ship in the same
APK** (the key lives in `registry.json` inside `assets/models/`). That is *obfuscation*,
not the §5.1 protection in `~/.claude/plans/frolicking-pondering-marble.md`:

- It defeats casual extraction (`unzip the APK, grab the .tflite` no longer works).
- It does *not* defeat a determined reverser who reads the bundled key and decrypts.

The proper defence — encrypted blob in TANUH's S3, key in `organisation_config` —
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
export tanuh_KEY_PASSWORD='…'
export tanuh_KEY_ALIAS='tanuh'
export OPENCHS_PROD_ADMIN_PASSWORD='…'    # the prod Avni admin password (per flavor_config.json)
```

## Per-build flow

1. **Drop the plaintext model in the source dir.** Anything under
   `tools/edge-model/source/*.tflite` is gitignored.

   ```bash
   cp /path/to/oral-cancer-v1.tflite tools/edge-model/source/
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

## What lives where

```
tools/edge-model/
├─ encrypt-model.js          # AES-GCM-256 encryption CLI (this dir)
├─ sample-override.json      # input/output contract for models without embedded metadata
├─ source/                   # plaintext .tflite files — gitignored
└─ README.md                 # this file

packages/openchs-android/android/app/src/tanuh/
├─ assets/models/            # encrypted blob + registry.json — gitignored, build-time only
├─ res/                      # branding (icons / splash) — replace generic placeholders with TANUH assets
└─ README.md
```

## When the model changes

Re-run `make tanuh-apk`. The encryption CLI generates a fresh AES key and IV per run,
so old encrypted blobs become invalid. This is intentional — there's no key-rotation
ambiguity, the build is reproducible from the plaintext source.

## Customising for vendor models with embedded metadata

If the `.tflite` carries TFLite Model Metadata (FlatBuffer, e.g. from Model Maker /
MediaPipe), you don't need `--override`. The native side reads input/output specs
straight from the embedded metadata. Just omit `--override` when invoking
`encrypt-model.js`.
