"""Stdlib mock regressions: synthetic files only, no model/network loads."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runtime import APP_ROOT, configure_runtime
configure_runtime()
import verify_real_runtime as verify


class OfficialPolicyConstantsTests(unittest.TestCase):
    def test_exact_audited_embedding_bin_identity_and_integrity(self):
        policy = verify.weight_policy("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")
        self.assertEqual(policy["name"], "pytorch_model.bin")
        self.assertEqual(policy["size"], 2271145830)
        self.assertEqual(policy["sha256"], "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38")
        self.assertTrue(policy["weights_only"])
        self.assertFalse(policy["trust_remote_code"])


class VerificationHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=APP_ROOT / ".cache/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bin_payload = b"synthetic embedding weight; NEVER a real model"
        # Replace only size/SHA expectations in mocks; keep real repo/revision policy gates.
        self.patches = [patch.object(verify, "APP_ROOT", self.root),
                        patch.object(verify, "EVIDENCE", self.root / "evidence"),
                        # File-policy tests fake Hub completely. Network bootstrap is tested in fresh processes.
                        patch.object(verify, "configure_runtime", wraps=lambda **kwargs: configure_runtime()),
                        patch.object(verify, "BGE_WEIGHT_BYTES", len(bin_payload)),
                        patch.object(verify, "BGE_WEIGHT_SHA256", hashlib.sha256(bin_payload).hexdigest())]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.payloads = {}
        self.infos = {}
        for repo, revision, architecture, weight in (
                (verify.BGE_MODEL_ID, verify.BGE_REVISION, "XLMRobertaModel", verify.BGE_WEIGHT_NAME),
                (verify.RERANK_MODEL_ID, verify.RERANK_REVISION, "XLMRobertaForSequenceClassification", "model.safetensors")):
            files = {name: name.encode() for name in verify.MODEL_SUPPORT_FILES}
            files["config.json"] = json.dumps({"model_type": "xlm-roberta", "architectures": [architecture]}).encode()
            files["tokenizer_config.json"] = json.dumps({"tokenizer_class": "XLMRobertaTokenizer"}).encode()
            files[weight] = bin_payload if repo == verify.BGE_MODEL_ID else b"synthetic safetensors"
            self.payloads[repo] = files
            self.infos[repo] = SimpleNamespace(sha=revision, siblings=[
                SimpleNamespace(rfilename=name, size=len(data), lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
                for name, data in files.items()])
        self.api = Mock(side_effect=lambda repo, **kwargs: self.infos[repo])
        self.download = Mock(side_effect=self.download_file)
        self.factory = Mock(return_value=SimpleNamespace(model_info=self.api))
        modules = patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(HfApi=self.factory, hf_hub_download=self.download)})
        modules.start()
        self.addCleanup(modules.stop)

    def snapshot(self, repo, revision):
        return self.root / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision

    def seed_snapshot(self, repo=verify.BGE_MODEL_ID, revision=verify.BGE_REVISION):
        snapshot = self.snapshot(repo, revision)
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, data in self.payloads[repo].items():
            (snapshot / name).write_bytes(data)
        return snapshot

    def download_file(self, repo, name, revision, cache_dir, local_files_only=False, **kwargs):
        self.assertEqual(Path(cache_dir), self.root / ".cache/huggingface/hub")
        path = self.snapshot(repo, revision) / name
        if local_files_only:
            if not path.is_file():
                raise FileNotFoundError("Synthetic cache miss; no network fallback")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.payloads[repo][name])
        return str(path)

    def test_provisions_exact_official_bin_and_only_safetensors_for_reranker(self):
        verify.provision()
        report = json.loads((self.root / "evidence/provisioning.json").read_text())
        self.assertTrue(report["both_models_provisioned"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.download.call_count, 12)
        for entry, expected_format in zip(report["models"], ("pytorch_bin", "safetensors")):
            self.assertEqual(entry["weight_format"], expected_format)
            self.assertTrue(entry["weight_sha256_verified"])
            self.assertEqual(entry["pinned_revision"], entry["revision"])
            self.assertFalse(entry["load_policy"]["trust_remote_code"])
        for call in self.download.call_args_list:
            repo, name = call.args
            expected_revision = verify.BGE_REVISION if repo == verify.BGE_MODEL_ID else verify.RERANK_REVISION
            self.assertEqual(call.kwargs["revision"], expected_revision)
            self.assertTrue(Path(call.kwargs["cache_dir"]).is_relative_to(self.root))
            if name.endswith(".bin"):
                self.assertEqual((repo, name), (verify.BGE_MODEL_ID, verify.BGE_WEIGHT_NAME))
        self.assertTrue(report["safetensors_conversion_disabled"])
        self.assertFalse(report["weights_converted"])
        self.assertEqual(os.environ["DISABLE_SAFETENSORS_CONVERSION"], "true")
        verify.configure_runtime.assert_called_with(offline=False)

    def test_rejects_unknown_model_or_unpinned_revision(self):
        for repo, revision in (("other/model", verify.BGE_REVISION), (verify.BGE_MODEL_ID, "main"),
                               (verify.RERANK_MODEL_ID, verify.BGE_REVISION)):
            with self.subTest(repo=repo, revision=revision), self.assertRaises(ValueError):
                verify.weight_policy(repo, revision)

    def test_reranker_bin_never_accepted_as_safetensors_replacement(self):
        info = self.infos[verify.RERANK_MODEL_ID]
        info.siblings = [file for file in info.siblings if file.rfilename != "model.safetensors"]
        info.siblings.append(SimpleNamespace(rfilename="pytorch_model.bin", size=10, lfs=None))
        with self.assertRaises(ValueError):
            verify.provision()
        self.assertFalse(any(call.args[0] == verify.RERANK_MODEL_ID for call in self.download.call_args_list))

    def test_rejects_revision_mismatch_before_download(self):
        self.api.side_effect = None
        self.api.return_value = SimpleNamespace(sha="unexpected")
        with self.assertRaises(ValueError):
            verify.provision()
        self.download.assert_not_called()

    def test_rejects_published_bin_size_or_hash_before_download(self):
        weight = self.infos[verify.BGE_MODEL_ID].siblings[-1]
        for field, bad in (("size", 0), ("lfs", SimpleNamespace(sha256="bad"))):
            with self.subTest(field=field), patch.object(weight, field, bad):
                with self.assertRaises(ValueError):
                    verify.provision()
        self.download.assert_not_called()

    def test_rejects_non_snapshot_cache_path_before_reading_weights(self):
        self.download.side_effect = None
        self.download.return_value = str(APP_ROOT / "README.md")
        with self.assertRaises(ValueError):
            verify.provision()

    def test_same_revision_in_wrong_model_snapshot_is_rejected(self):
        cache = verify.owned_cache()
        wrong = cache / "models--other--model/snapshots" / verify.BGE_REVISION / verify.BGE_WEIGHT_NAME
        with self.assertRaises(ValueError):
            verify.validate_snapshot_path(wrong, cache, verify.BGE_MODEL_ID, verify.BGE_REVISION, verify.BGE_WEIGHT_NAME)

    def test_redirected_cache_directory_is_rejected(self):
        with patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            verify.owned_cache()

    def test_local_load_verifies_file_and_uses_unchanged_canonical_helper_offline(self):
        self.seed_snapshot()
        canonical = Mock(return_value=("tokenizer", "model"))
        with patch.dict(sys.modules, {
                "search_logic": SimpleNamespace(MODEL_ID=verify.BGE_MODEL_ID, MODEL_REVISION=verify.BGE_REVISION),
                "generar_embeddings_bge_m3_colab": SimpleNamespace(load_model=canonical)}):
            tokenizer, model, provenance = verify.load_verified_embedding_model("cpu", "fp32", 8192)
        canonical.assert_called_once_with("cpu", "fp32", 8192, local_files_only=True)
        self.assertEqual((tokenizer, model), ("tokenizer", "model"))
        self.assertTrue(self.download.call_args.kwargs["local_files_only"])
        self.factory.assert_not_called()
        self.assertTrue(provenance["sha256_verified"])
        self.assertEqual(provenance["format"], "pytorch_bin")
        self.assertFalse(provenance["trust_remote_code"])
        self.assertTrue(provenance["weights_only"])
        self.assertEqual(os.environ["DISABLE_SAFETENSORS_CONVERSION"], "true")

    def test_wrong_local_size_or_hash_prevents_model_load(self):
        snapshot = self.seed_snapshot()
        weight = snapshot / verify.BGE_WEIGHT_NAME
        valid = weight.read_bytes()
        for data in (b"wrong-size", b"X" * len(valid)):
            with self.subTest(data=data):
                weight.write_bytes(data)
                with patch.object(verify, "_canonical_embedding_load") as loader:
                    with self.assertRaises(ValueError):
                        verify.load_verified_embedding_model("cpu", "fp32", 8192)
                    loader.assert_not_called()

    def test_converted_embedding_weights_prevent_model_load(self):
        snapshot = self.seed_snapshot()
        (snapshot / "model.safetensors").write_bytes(b"forbidden conversion")
        with patch.object(verify, "_canonical_embedding_load") as loader, self.assertRaises(ValueError):
            verify.load_verified_embedding_model("cpu", "fp32", 8192)
        loader.assert_not_called()

    def test_remote_code_configuration_prevents_model_load(self):
        snapshot = self.seed_snapshot()
        path = snapshot / "tokenizer_config.json"
        path.write_text(json.dumps({"tokenizer_class": "XLMRobertaTokenizer", "auto_map": {"AutoTokenizer": "remote.Code"}}))
        with patch.object(verify, "_canonical_embedding_load") as loader, self.assertRaises(ValueError):
            verify.load_verified_embedding_model("cpu", "fp32", 8192)
        loader.assert_not_called()

    def test_canonical_helper_identity_mismatch_prevents_loading(self):
        canonical = Mock()
        with patch.dict(sys.modules, {"search_logic": SimpleNamespace(MODEL_ID="other/model", MODEL_REVISION=verify.BGE_REVISION),
                                     "generar_embeddings_bge_m3_colab": SimpleNamespace(load_model=canonical)}):
            with self.assertRaises(ValueError):
                verify._canonical_embedding_load("cpu", "fp32", 8192)
        canonical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
