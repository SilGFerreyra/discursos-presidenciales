# Independent presidential-speeches app with local reranker

**PASS for the automated core: ordinary production startup, real cached query,
default K=10/N=25 reranking, navigation, and export payloads are verified.** Fresh
processes inherited offline flags `0`; production itself set them before imports,
and real-model guards observed zero external network attempts. No verifier offline
flag workaround was used. Browser row-click/download remains manual pending, and
relevance improvement is not claimed without human labels.

## Start locally

Source/model caches stay in this H: `app_reranker` directory. The authorized Python is:

`C:\Users\Usuario\AppData\Local\app-reranker\.venv\Scripts\python.exe`

Use `ejecutar_app.bat`, or `ejecutar_app.bat --check` to inspect the environment
without starting a server. The launcher chooses an unused localhost port starting
at 8502 and never stops another process or installs packages.

Production startup sets **HF_HUB_OFFLINE=1**, **TRANSFORMERS_OFFLINE=1** and
**DISABLE_SAFETENSORS_CONVERSION=true** before relevant imports, including direct
script/resource entry. Inherited online flags are overridden. If another caller
already imported HF/Transformers online, startup fails closed and requires a fresh
process; it never mutates the libraries' cached globals. The prior per-call offline
checkbox was removed because it could not enforce import-time behavior.

1. Enter four carriers and press **Analizar cuatro carriers**.
2. Enable the reranker and press **Aplicar reranker** explicitly.
3. Inspect cosine and raw logits separately; exports contain all final K rows.

Filters/pages/selections/toggles do not infer. Draft carriers do not replace the
analyzed query. Failures invalidate stale scores and keep truthful cosine fallback.
Missing models cause a visible local-cache error, never an automatic download.

## Conservative CPU preset, not a fast mode

Initial **K=10 / N=25**, with initial-N formula `max(25, 2*K)`; upper limits remain
K=500/N=1000. Values are editable. Existing/manual N is not silently clamped when
K changes: N<K is an error and disables Apply. N=25 means **100** carrier/passage pairs.

The independent real cosine benchmark used CPU FP32, four threads, batch 4,
max 512 tokens, with both models resident:

| N | Pairs | Measured inference |
|---|---|---|
| 25 | 100 | 169.01 s |
| 50 | 200 | 204.52 s |
| 100 | 400 | 407.27 s |
| 200 | 800 | 786.07 s (13.10 min) |

Peak sampled RSS: **4,010,971,136 bytes (~4 GB)**. The full suite took **26.78 min**.
Separate offline N=25: **105.14 s**; education N=25: **115.04 s**. Text length, first
forward/cache state and system load vary; these are measured examples, **not an
ETA guarantee**, linear extrapolation or N=1000 recommendation. Earlier lexical
reranker-only smoke (127.90 s) is separate historical evidence.

## Provisioning is a separate online process

Both models are present; **no provisioning is needed for ordinary startup**.
If explicitly restoring/provisioning them later, close the app and use a fresh process:

```powershell
$py = 'C:\Users\Usuario\AppData\Local\app-reranker\.venv\Scripts\python.exe'
& $py -B verify_real_runtime.py provision
```

Only this command bootstraps online (`offline=False`) before imports; conversion
stays disabled. All verification loads/smokes are offline. The benchmark's legacy
`--allow-download` switch now rejects online loading and directs users to provisioning.

Formats/pins are unchanged: only official `BAAI/bge-m3` pin
`5617a9f61b028005a4858fdac845db406aefb181` may use its audited bin
(2,271,145,830 bytes, SHA256
`b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`).
The unchanged canonical helper retains restricted `weights_only=True`; the new
reranker pin `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` requires safetensors and
`trust_remote_code=False`. No conversion, revision substitution or pooling changes.

## Verification and remaining scope

From this directory:

```powershell
& $py -B -m unittest test_isolation test_streamlit_search test_rerank_logic test_streamlit_rerank test_verify_real_runtime test_resume_verification test_offline_startup test_artifact_integrity
& $py -I -B -m pip --isolated --disable-pip-version-check check
& $py -I -B inventory.py hashes
```

Fresh-process tests cover direct resources, launcher validation, direct app script,
early flags, rejecting late switches/cache-hit online arguments, and intentionally
online provisioning bootstrap with no network/model downloads. A lightweight probe
also used the **actual cached tokenizer with model weights mocked**, observing zero
HTTP/socket attempts. It is not actual carrier inference or full-model UI proof.
The writer's latest suite passed **97 tests** (including real read-only corpus
integrity), preserved as historical execution rather than rerun by the focused
verifier. Fresh `pip check`, 26-source syntax/import checks, ordinary Python/batch
launcher checks and real query/UI/HTTP verification passed.

### Final ordinary-startup proof

The existing launcher-entry query probe loaded actual canonical weights/tokenizer,
embedded four real default carriers and rejected an online argument even on a
resource-cache hit. It recorded **zero HTTP/socket attempts**. The focused real UI
observer installed guards first, then imported normal production resources; it
does not call `configure_runtime` or write offline flags. Actual direct app rendering
used untouched default controls: K=10, N=25, page size 25, batch 4, max 512, CPU,
with the misleading offline checkbox absent.

Nine snapshots verified analysis, pending, explicit Apply, K/page reuse,
filter-pending/page reset, reapply, disable and a cheap rerun. Exactly **one UI
query-embedding call** and **two explicit N=25 calls**, each 100 finite carrier pairs
and zero truncations; scoring took **117.97 s / 102.42 s** with four test CPU threads,
FP32. The UI flow took **325.02 s** including loading/rendering. Finite real cosine
matrix shape **87,546 × 5**, four-logit means/order, rendered IDs/detail and all final
CSV/JSON/TXT rows/state/method were checked. Payload checks are not browser clicks.

Reproduce the focused proof with inherited online flags, not a strict-mode workaround:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '0'
$env:TRANSFORMERS_OFFLINE = '0'
& $py -B offline_startup_probe.py --entry launcher --real-model --output .cache/evaluation/offline-startup-real-query-probe.json
& $py -B final_verify_runtime.py ui
```

Ordinary Python and batch `--check` both passed from those inherited flags. The
ordinary launcher server also passed health/root HTTP 200 at port 8502, then only
its created process tree was stopped to avoid duplicate models; no verification
server is left running. Start normally with `ejecutar_app.bat`; no extra offline
environment configuration or model download is needed.

Fresh evidence: `.cache/evaluation/offline-startup-real-query-probe.json`,
`final-default-ui.json`, `final-ui-*.csv/json/txt`, `final-default-server.json`,
`final-default-server.log`. The old failing probe remains historical, now resolved
by the writer and verified with actual weights. No production code was edited by
this verifier. The previous 26.78-minute benchmark/two concepts were **not rerun**.

Browser row-click/download checks remain manual; precision@10/nDCG@10 are
**not measured** without human labels. Existing `.cache/evaluation/benchmark*.json`,
offline/education runs, ranking comparisons, UI payloads and the old failing
`default-local-query-probe.json` are preserved. See `verification-status.json` for
the new source/test status and `real-verification-report.md` for historical evidence.

Original app/helper/parent tests/corpus remain read-only. This batch checks the
27 protected SHA256+size/mtime entries against the preserved original baseline;
no redundant full 42k traversal or all-byte proof is claimed. C: owns only the
authorized venv/installer cache; H: owns runtime/model caches. The partial H: venv
was not executed or repaired. The writer batch ran no server/real weights; final
focused verifier execution above is explicitly separate.
