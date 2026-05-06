# tanuh.mk — build automation for the TANUH HealthForge edge-model flavour.
#
# Two-command flow on a TANUH developer's machine:
#   make tanuh-setup     # one-time: generate signing keystore
#   make tanuh-apk       # per-build: encrypt model → assemble signed APK
#
# Plaintext model: tools/edge-model/source/<TANUH_MODEL_KEY>.tflite (gitignored)
# Encrypted output: packages/openchs-android/android/app/src/tanuh/assets/models/
#                   {<TANUH_MODEL_KEY>.bin, registry.json}    (gitignored)
# Signed APK:       packages/openchs-android/android/app/build/outputs/apk/tanuh/release/app-tanuh-release.apk
#
# See tools/edge-model/README.md for the full documentation.

# Override on the command line: TANUH_MODEL_KEY=oral-cancer-v2 make tanuh-apk
TANUH_MODEL_KEY ?= oral-cancer-v1
TANUH_MODEL_SRC := tools/edge-model/source/$(TANUH_MODEL_KEY).tflite
TANUH_OUT_DIR   := packages/openchs-android/android/app/src/tanuh/assets/models
TANUH_OVERRIDE  := tools/edge-model/sample-override.json
TANUH_KEYSTORE  := tanuh-release-key.keystore

tanuh-setup: ## One-time: generate the tanuh release keystore.
	@if [ -f "$(TANUH_KEYSTORE)" ]; then \
		echo "Keystore already exists at $(TANUH_KEYSTORE) — skipping. Delete it first to regenerate."; \
		exit 0; \
	fi
	@echo "Generating tanuh release keystore. You will be prompted for passwords and DN."
	@keytool -genkeypair -v \
		-keystore $(TANUH_KEYSTORE) \
		-alias tanuh \
		-keyalg RSA -keysize 2048 \
		-validity 10000
	@echo ""
	@echo "Keystore created. Set these env vars in your shell before running 'make tanuh-apk':"
	@echo "  export tanuh_KEYSTORE_PASSWORD='<the keystore password you just chose>'"
	@echo "  export tanuh_KEY_PASSWORD='<the key password you just chose>'"
	@echo "  export tanuh_KEY_ALIAS='tanuh'"

tanuh-encrypt: ## Encrypt the plaintext model and emit registry.json.
	@if [ ! -f "$(TANUH_MODEL_SRC)" ]; then \
		echo "ERROR: $(TANUH_MODEL_SRC) not found."; \
		echo "Drop your plaintext .tflite there (filename = TANUH_MODEL_KEY + .tflite)."; \
		echo "  cp /path/to/$(TANUH_MODEL_KEY).tflite tools/edge-model/source/"; \
		exit 1; \
	fi
	@mkdir -p $(TANUH_OUT_DIR)
	node tools/edge-model/encrypt-model.js \
		--in $(TANUH_MODEL_SRC) \
		--out-dir $(TANUH_OUT_DIR) \
		--model-key $(TANUH_MODEL_KEY) \
		--override $(TANUH_OVERRIDE) \
		--default-model

tanuh-apk: tanuh-encrypt ## Encrypt model + assemble signed tanuh release APK.
	@if [ ! -f "$(TANUH_KEYSTORE)" ]; then \
		echo "ERROR: $(TANUH_KEYSTORE) not found. Run 'make tanuh-setup' first."; \
		exit 1; \
	fi
	@if [ -z "$$tanuh_KEYSTORE_PASSWORD" ] || [ -z "$$tanuh_KEY_PASSWORD" ] || [ -z "$$tanuh_KEY_ALIAS" ]; then \
		echo "ERROR: signing env vars not set. Export tanuh_KEYSTORE_PASSWORD, tanuh_KEY_PASSWORD, tanuh_KEY_ALIAS."; \
		exit 1; \
	fi
	cd packages/openchs-android/android; GRADLE_OPTS="$(if $(GRADLE_OPTS),$(GRADLE_OPTS),-Xmx1024m -Xms1024m)" ./gradlew assembleTanuhRelease --stacktrace
	@echo ""
	@echo "Signed APK: packages/openchs-android/android/app/build/outputs/apk/tanuh/release/app-tanuh-release.apk"

tanuh-clean: ## Remove the per-build encrypted blob and registry.json.
	rm -f $(TANUH_OUT_DIR)/*.bin $(TANUH_OUT_DIR)/registry.json
	@echo "Cleared $(TANUH_OUT_DIR)/"

.PHONY: tanuh-setup tanuh-encrypt tanuh-apk tanuh-clean
