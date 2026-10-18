# Final verification: ordinary startup and real default UI PASS

## Final Verification Report

**Status: success. Automated core verdict: PASS. Overall: PASS WITH WARNINGS** for
manual browser interactions, CPU latency and unmeasured human relevance. The prior
default-loader metadata-HTTP finding is closed by the writer's production fix and
fresh real ordinary-startup execution. No verifier production fix or offline-mode
workaround was used.

**Change/contract:** independent `app_reranker`; approved plan
`C:\Users\Usuario\.local\share\opencode\plans\1790797894034-proud-knight.md`, latest
`verification-status.json`, and authorized writer startup/K10-N25 adjustment.
Standard verification, `artifact_store.mode=none`, no new SDD pipeline, delegation,
commits/PRs, installs, original edits or strict-TDD claim. Formal spec/design/task
artifacts are absent: formal delta-spec/design coherence/checked-task counts are
skipped, not fabricated. Technical proof follows the supplied runtime contract.

### Completion and runtime compliance

| Required behavior | Actual executed evidence | Result |
|---|---|---|
| Ordinary launcher overrides inherited online flags before imports | Existing launcher-entry probe with real weights, initial HF/TF flags `0`, actual early-import observations `1` | PASS |
| Actual unchanged canonical model embeds real four carriers | Real cached weights/tokenizer, four `ok` flags, no fake weights or tokenizer | PASS |
| Direct production resources/app startup, no verifier flag forcing | New focused observer installs guards, then imports ordinary `app_resources`; never invokes bootstrap or sets offline flags | PASS |
| Default controls K10/N25 and removed offline checkbox | Actual AppTest rendered defaults before any control adjustment | PASS |
| Finite cosine matrix and four logits per candidate | Real matrix `(87546,5)`/vectors `(4,1024)`, actual N25 `(25,4)` outputs, means/order/cosine preservation assertions | PASS |
| Explicit inference only, candidate/pagination state, exports | Nine real workflow snapshots, one UI embedding call/two explicit scoring calls | PASS |
| Both ordinary launcher checks and actual HTTP startup | Python/batch checks from flags `0`; free8502 health/root200; controlled stop | PASS |
| Historical pinned provisioning/complete CPU/two-concept evidence | Original real evidence retained unchanged; expensive suite not repeated | RETAINED PASS |
| Protected original source/config checks | Fresh final27 SHA256+size/mtime versus preserved baseline; zero changes; no full42k repeat | PASS within declared scope |
| Browser row click/actual downloads; human relevance | AppTest cannot simulate row click; no human labels | MANUAL PENDING / NOT MEASURED |

### No hidden strict-mode workaround

Every focused query/UI/launcher/server command started with
`HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0`. The verifier did not pre-set `1`.
Production launcher/resources/script bootstrap set `1` before actual HF/Transformers
imports, and their cached modes were observed true. Conversion suppression remained
`DISABLE_SAFETENSORS_CONVERSION=true`. The old `resume_verification.py` helper was
not used for this proof because it independently sets offline flags.

The new `final_verify_runtime.py` is a standalone observer only. Requests and
non-loopback socket guards were installed before production imports/model calls.
Numeric localhost socketpairs required by Windows asyncio were separately recorded;
they are not remote API attempts. Both real launcher-query and UI proof recorded
**zero external attempts**, not merely zero allowed traffic. No HF/Transformers
cached globals were patched. The writer's fail-closed late-online-import behavior
and cache-hit guards remain supported by the preserved 97-test suite.

### Actual models, query, defaults and CPU execution

Own Python: `C:\Users\Usuario\AppData\Local\app-reranker\.venv\Scripts\python.exe`;
Python3.12.6, Torch2.14.1+cpu, Transformers4.57.6, Hub0.36.2, Streamlit1.64.0.
`-B`/`PYTHONDONTWRITEBYTECODE=1`; NEWAPP H: HF/Torch/MPL/TEMP caches; parent canonical
helper/corpus/tests and original application/environment read-only.

Model cache root:
`H:\Mi unidad\Silvana\discursos presidenciales\app nueva\app_reranker\.cache\huggingface\hub`

- Canonical BGE-M3: `models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181\pytorch_model.bin`;
  2,271,145,830 bytes; previously verified SHA256
  `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`.
- Reranker: `models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\model.safetensors`;
  2,271,071,852 bytes; previously verified SHA256
  `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`.

Actual query/reranker model config pins, FP32/eval mode were asserted in the UI
observers. Actual canonical-bin `torch.load(weights_only=True)` calls were observed
on the exact NEWAPP snapshot with the correct size. Model outputs were real, not
mocked; the unchanged canonical helper remains the query implementation. No
conversion, alternate model, unpinning, package or production logic changes.
Prior actual full hashes/provisioning provenance are retained; no unnecessary
large-weight rehash or download was repeated.

Displayed initial controls: **K10/N25, page25, batch4, max512, CPU**, reranker off
until enabled/applied, offline checkbox absent. Four CPU threads were an explicitly
declared test-measurement setting, not a claim that the launcher forces that count.

| Fresh focused metric | Actual value |
|---|---|
| UI carrier inference | 10.8034 s, one call/four real carriers, flags all `ok` |
| First explicit N25 rerank | 117.9667 s; 100 finite pairs, zero truncations |
| Country-filter reapply N25 | 102.4203 s; 100 finite pairs, zero truncations |
| Full focused UI flow | 325.0246 s including imports/load/rendering |
| Network attempts | Query probe 0; UI external HTTP/socket attempts 0 |

Separate existing launcher-query probe also made one actual embedding call in its
own fresh process. Within the UI flow, there was **exactly one** embedding call;
cheap reruns did not re-encode carriers or implicitly score candidates.

Workflow: analysis → pending → Apply K10/N25 → K20 score reuse → page2 → Argentina
filter/pending/page1 reset → reapply K10 → disable → cheap rerun. K10/page10 has one
page, so page2 used K20 within unchanged N25; manual N stayed25, not clamped. All
full final CSV/TXT rows and JSON method/state/subset matched the actual final ranking.
Rendered table IDs, default fragment detail, cosines and raw-logit means/order were
asserted. Saved export payloads are **not** browser download-click evidence.

### Exact executed commands and test scope

Working directory for all commands:
`H:\Mi unidad\Silvana\discursos presidenciales\app nueva\app_reranker`.
`$py` denotes the exact own Python above. Before each ordinary-startup command:
`$env:PYTHONDONTWRITEBYTECODE='1'; $env:HF_HUB_OFFLINE='0'; $env:TRANSFORMERS_OFFLINE='0'`.

| Command | Result |
|---|---|
| `& $py -B offline_startup_probe.py --help` | Actual CLI read before execution |
| Own Streamlit discovery script, `--project-dir C:\Users\Usuario\AppData\Local\app-reranker` | Installed1.64 testing/performance/CLI docs read |
| `& $py -B offline_startup_probe.py --entry launcher --real-model --output .cache/evaluation/offline-startup-real-query-probe.json` | PASS, actual weights/carriers, early offline capture, zero attempts, online cache-hit request rejected |
| `& $py -B final_verify_runtime.py ui` | PASS, bounded1,200,000ms; actual default controls/full query matrix/reranking/export flow |
| `& $py -B launch.py --check` | PASS, exact own interpreter/prefix/user-site false/cache paths/network policy |
| `cmd.exe /d /c 'ejecutar_app.bat --check'` | PASS, same ordinary production startup |
| `& $py -I -B -m pip --isolated --disable-pip-version-check check` | No broken requirements found |
| `& $py -B final_verify_runtime.py server` | PASS, actual ordinary launcher HTTP health/root; no verifier offline parameters |
| `& $py -I -B verify_sources.py` | 26 Python source syntax/import-boundary checks,97 existing definitions; not a fresh97-test suite |
| `& $py -I -B inventory.py hashes` | Fresh final protected-source/config hash check; scope below |
| `& $py -B final_verify_runtime.py check` / `& $py -B verify_real_runtime.py check-evidence` | Final fresh proof/status/docs/hashes consistency PASS; historical evidence remains separately labeled |

**Tests:** latest writer **97 passed / 0 errors / 0 failures** in292.621s,
preserved as historical execution, not rerun or relabeled by the verifier. Current
actual runtime assertions are separate executed proof, not additional unittest
counts. Build: syntax/import checks passed; no separate build/coverage percentage
measured. No strict-TDD claim. No 26.78-minute benchmark rerun.

### Server status, invariance and retained evidence

Ordinary own launcher chose free **8502**, URL **http://127.0.0.1:8502**;
health **200 `ok`**, root **200 / 7,260 bytes**. Created/stopped PIDs:
**20308,16396,9872,11524**. Only that created process tree was stopped after
ancestry/command ownership checks and waits. **Stopped explicitly at return**;
no original/other process was terminated and no duplicate model resources left.
Start with ordinary `ejecutar_app.bat`; no external strict-mode flags are needed.

Original baseline remains
`9c79e1c9bacf6e563536f95f3dbb953f18b468f849be6180f12bb0330c519236`.
Fresh final invariance check found **zero changes across 27 selected original
source/config SHA256 plus size/mtime**, not the entire dependency payload/corpus or system. The historical
completed42,643-entry report is retained, not relabeled current. Known H: full-tree
timeouts were not repeated. The partial H: new venv was not executed/repaired or
written by verifier operations; old app/canonical/corpus remained read-only.

Fresh proof files: `offline-startup-real-query-probe.json`, `final-default-ui.json`,
`final-ui-*.csv/json/txt`, `final-default-server.json`, `final-default-server.log`
under `.cache/evaluation/`. Original benchmark/education/offline/comparison files
and historical failing probes are preserved. Earlier 127.90s lexical smoke is not
substituted for the real cosine evidence.

### Issues, final verdict and next recommendation

**CRITICAL: none in the verified automated core.** Prior ordinary-startup metadata
request issue is resolved by writer and independently proven with real weights.
**WARNING:** CPU N25 still takes roughly two minutes; browser row-click/actual
download remains manual pending; precision@10/nDCG@10 **not measured** without
human labels. Semantic/logit ranking changes are not proof of better relevance.
**Next: none (no new SDD phase or production correction).** Perform manual browser
checks and human relevance judgments when appropriate. **Automated core PASS;
overall PASS WITH WARNINGS** for those deliberately unclaimed dimensions.

**Skill resolution: paths-injected**, exact requested sdd-verify,
developing-with-streamlit and cognitive-doc-design skill files reread; shared SDD
protocol and installed own-environment Streamlit1.64 testing/performance/CLI docs.

---

## Historical resumed report — prior finding now resolved, retained for audit

The report below is pre-writer-fix evidence, **not the current verdict**. Its
ordinary-startup finding is closed by the final real proof above; original failure
artifacts remain available. Full benchmark metrics are retained without rerunning.

## Current Verification Report

**Status: partial. Verdict: FAIL for the unqualified default local-only/offline
contract.** Both official model pins are provisioned and their real bytes verified;
all requested CPU budgets, separate offline reload/inference, two concept
comparisons, and the real-model UI/export workflow completed. These passed with
offline flags set before imports. A fresh probe of the actual app query loader
without those process flags confirmed a metadata HTTP attempt despite
`_local_files_only=True`. The verifier reports this to the writer, not fixes it.

**Contract/mode:** approved Spanish plan and `verification-status.json`, standard
verification, `artifact_store.mode=none`. No new SDD pipeline, delegation, commits,
PRs, installs, original edits, or production logic edits. No strict-TDD or coverage
percentage claim. Formal spec/design/task-checklist artifacts are absent, so formal
delta-spec/design coherence and checked-task counts are skipped rather than invented.

### Completeness and behavioral compliance

| Contract item | Actual covering execution | Result |
|---|---|---|
| Both official pins/weight formats | Corrected `verify_real_runtime.py provision`, actual size/SHA256 checks | PASS |
| Restricted unchanged canonical bin load | `embedding-load` invoked through `resume_verification.py embedding`; actual `torch.load(weights_only=True)` observations | PASS |
| True cosine N=25/50/100/200, both models loaded | Unchanged benchmark helper invoked with real delegating observers and progress/checkpoints | PASS |
| Separate strict offline reload/inference | `offline-benchmark.json`, N=25 IDs/logits reproduce primary N=25 | PASS |
| Two Spanish concepts/top-ten comparisons | Electoral corruption N=200 and education access N=25; full aligned records | PASS for comparison evidence, not relevance |
| Real UI workflow/cache/state/exports | Nine AppTest snapshots, actual corpus/weights and payload checks | PASS under strict startup |
| Own launcher HTTP startup | Free 8502, health/root 200, only created tree stopped | PASS |
| Default local-cache toggle prevents API requests | Actual `app_resources.load_query_model_cached(...,_local_files_only=True)` with default process flags; blocked metadata GET | **FAIL** |
| Fresh protected-source hashes | 27 SHA256 + size/mtime against original baseline, zero changes | PASS within that scope |
| Fresh full 42k inventory | 240 s timeout; no remaining inventory process; prior full report retained as historical | INCOMPLETE; not an all-byte proof |
| Browser row click/download | No existing browser tool/package available; no install | MANUAL PENDING |
| Human relevance | No labels; precision@10/nDCG@10 not measured | NOT MEASURED, not a runtime blocker |

### Provisioning and provenance

Both snapshots are inside:
`H:\Mi unidad\Silvana\discursos presidenciales\app nueva\app_reranker\.cache\huggingface\hub`

| Model | Exact pin | Weight file, size, SHA256 |
|---|---|---|
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | `models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181\pytorch_model.bin`; 2,271,145,830 bytes; `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38` |
| BAAI/bge-reranker-v2-m3 | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\model.safetensors`; 2,271,071,852 bytes; `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286` |

Twelve required files passed size checks; actual weight hashes match both audited
values and official LFS metadata. Provisioning took **96.3019 s**, without loading
models. No global-cache writes/copies, conversion, unpinning or replacement.
NEWAPP sets `DISABLE_SAFETENSORS_CONVERSION=true` process-locally. BGE-M3 was then
rehash-verified and loaded by the unchanged parent helper with strict local flags:
**13.4224 s** including integrity/load, with actual restricted bin loads observed.
Reranker retains safetensors-only and `trust_remote_code=False`. Real benchmark
observers asserted both model config commits, FP32 and eval mode.

### CPU metrics: real cosine candidates, both models loaded

CPU FP32, four Torch intra-op threads, batch 4, max pair length 512 including
special tokens. The unchanged canonical helper embeds the actual four carriers;
cosine selection uses the real complete read-only corpus (87,546 chunks). No
lexical or simulated candidate substitutes.

| N | Pairs | Inference seconds | Minutes | Truncated pairs |
|---|---|---|---|---|
| 25 | 100 | 169.0058 | 2.82 | 0 |
| 50 | 200 | 204.5210 | 3.41 | 0 |
| 100 | 400 | 407.2742 | 6.79 | 0 |
| 200 | 800 | 786.0724 | 13.10 | 0 |

Suite: **1,606.7742 s (26.78 min)**, completed within the 2,400,000 ms bound.
Local cold-process load: BGE-M3 **6.2186 s**, reranker **1.3800 s**. Reranker-only
offline reload with BGE-M3 still loaded: **1.0934 s**, followed by actual four-pair
smoke inference **10.7965 s**. Memory sampled every 50 ms: RSS peak
**4,010,971,136 bytes**; OS peak working set **4,011,892,736 bytes**. RSS is sampled
physical memory, not a guarantee of all virtual/committed memory or other-process
consumption. Both model objects remained loaded during scored runs.

Separate strict offline default N=25: inference **105.1423 s**, total **144.6659 s**;
education-access N=25: inference **115.0439 s**, total **153.5962 s**. Both also
actually reloaded the reranker locally and scored one passage/four pairs. All
successful inference runs recorded **zero external HTTP/socket attempts** and
finite four-logit arrays for every candidate. No-network flags were set before
dependency imports; bin-load observers delegated unchanged restricted calls.

`ranking-comparisons.json` aligns the union of each cosine/reranker top ten by
embedding-row identity, with full text/metadata, cosine scores, four raw logits,
means, retrieval/reranker ranks and times. Electoral top-ten overlap: **5/10**;
education: **6/10**. Overlap is ranking-change evidence, **not** relevance quality.
Precision@10/nDCG@10 are **not measured**. The previous 127.90 s lexical smoke
remains historical and is not merged into these measurements.

**Writer recommendation:** offer an explicit initial CPU K=10/N=25 preset. It still
took approximately 99–169 s across real runs on this machine; K=100/N=200 is costly
at 13.10 min. First-forward/cache/system-load effects make the runs non-identical
latency conditions. Do not extrapolate N=1000 or silently alter the approved
`max(100,2*K)` default policy; production settings remain unchanged.

### Actual AppTest and server

Workflow: analysis → enable/pending → Apply K=10/N=25 → K=20 with unchanged
candidate scores → page 2 → country filter/pending and page reset → reapply K=10
→ disable → cheap rerun. K=10 at minimum page size 10 has one page; K=20 was used
only to exercise actual pagination while reusing the same N=25 scores.

Exactly **one** actual carrier-embedding call and **two** explicit actual reranker
calls. Score times: **118.1413 s**, **98.7431 s**, each 25 candidates/100 finite
pairs/zero truncations. Rendered dataframe IDs/order and default fragment details
were asserted; all final CSV/TXT rows and JSON state/method metadata matched the
complete final ranking, not only the page. Payloads are saved as `ui-*.csv/json/txt`.
This does not simulate clicking the download buttons or dataframe rows in a browser.
Windows asyncio required **14 internal numeric-loopback socketpair connections**;
only that exception was allowed, with all HTTP and non-loopback connects rejected.

Actual own launcher: free **8502**, health **200 `ok`**, root **200 / 7,260 bytes**.
Created and stopped PIDs: **14824, 11624, 16312, 19024**. Process ancestry/command
ownership and termination waits were checked. **Server stopped explicitly at
return**; no original/unrelated PID stopped, no duplicate models left by verifier.

### Confirmed issue for writer

**CRITICAL for the advertised offline contract:** without process-level offline
flags, the real app query resource loader called with `_local_files_only=True`
attempts `GET https://huggingface.co/api/models/BAAI/bge-m3`. The probe had both
cached offline modes false, matching ordinary launcher startup; the rejecting
HTTP observer stopped it before traffic or model inference. The request concerns
model metadata only; no carrier/passage transmission was observed.

Installed Transformers 4.57.6 `tokenization_utils_base.py:2431–2442` calls Hub
`model_info` during tokenizer Mistral-regex detection without honoring that
local-only argument, unless import-cached offline mode is active. Production
`app_resources.py:31–38` delegates to the unchanged canonical helper and ordinary
`launch.py` does not itself set these offline flags. No production fix was made.
Evidence: `.cache/evaluation/default-local-query-probe.json`, including traceback.

README documents the verified strict startup (set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` before launching Python). This makes cached search usable,
but does not repair the ordinary toggle/launcher contract. Writer should address
that guarantee without modifying the parent helper/libraries or breaking the
separate explicit provisioning path; then rerun the targeted no-network test.

### Commands, test/build evidence and retained incidents

All commands used own Python
`C:\Users\Usuario\AppData\Local\app-reranker\.venv\Scripts\python.exe`, `-B` /
`PYTHONDONTWRITEBYTECODE=1`, and NEWAPP H: runtime caches. Active versions remain
Python 3.12.6, Torch 2.14.1+cpu, Transformers 4.57.6, Hub 0.36.2, Streamlit 1.64.0.

| Command suffix | Result |
|---|---|
| `-B verify_real_runtime.py --help` and `-B benchmark_reranker.py --help` | Actual CLI confirmed; no invented `provision-embedding` command |
| Own Streamlit discovery script with C: own project | Installed 1.64 skill/testing/performance/CLI references read |
| `-B resume_verification.py prepare` | Historical docs/status/provision/server preserved in `before-resume/` |
| `-B verify_real_runtime.py provision` | Passed, 900,000 ms bound |
| `-B resume_verification.py embedding` | Passed, actual unchanged local embedding-load path |
| `-B resume_verification.py benchmark --budgets 25,50,100,200 --output .cache/evaluation/benchmark.json` | All budgets passed, 2,400,000 ms bound |
| `-B resume_verification.py benchmark --budgets 25 --carriers-file .cache/evaluation/education-carriers.json --output .cache/evaluation/education-benchmark.json` | Passed |
| `-B resume_verification.py benchmark --budgets 25 --output .cache/evaluation/offline-benchmark.json` | Passed, separate process/reload/inference |
| `-B resume_verification.py ui` | Actual full workflow passed, 1,200,000 ms bound |
| `-B verify_real_runtime.py server` | Actual own launcher health/root passed, controlled stop |
| `-B resume_verification.py summarize` | Complete candidates/finite logits, aligned comparison, offline reproducibility and UI counts validated |
| `-B verify_real_runtime.py check-evidence` | Final docs/status/model JSON consistency passed; includes historical smoke/inventory evidence without relabeling it fresh |
| `-B resume_verification.py plain-local-probe` | Default-loader finding confirmed; HTTP stopped before traffic |
| `-I -B -m pip --isolated --disable-pip-version-check check` | No broken requirements found |
| `-B -m unittest -v test_resume_verification` | **5 passed**, final 27.015 s; observer guards only |
| `-I -B verify_sources.py` | 23 sources compile/parse/import boundaries passed; 87 definitions, not 87 freshly executed tests |
| `-I -B inventory.py hashes` | Fresh final 27 hashes + size/mtime unchanged |
| `-I -B inventory.py after` | 240 s timeout, no remaining inventory process; no retry or environment workaround |

**Historical writer tests:** 38 targeted/82 full tests passed and remain preserved;
the 82 suite was not rerun here. No build beyond syntax/import checks or coverage
percentage measured. Passive delegating observers do not replace scores, metadata,
models, query embeddings, corpus or app rendering.

Two verifier-only guard incidents were corrected and preserved: first benchmark
stopped before inference because offline flags were set after imports (8.4145 s,
one blocked metadata GET); first UI attempt stopped before app start because the
socket guard also blocked Windows asyncio loopback. These are not library installs,
production fixes, model substitutions, benchmark timeouts or automatic retry loops.
Artifacts: `benchmark-checkpoint-failed-before-resume.json` and
`real-ui-workflow-guard-failure.json`; later successful evidence is distinct.

### Invariance, limitations and next recommendation

Fresh final **27 selected source/config hashes plus size/mtime: zero changes**;
baseline SHA256 remains
`9c79e1c9bacf6e563536f95f3dbb953f18b468f849be6180f12bb0330c519236`.
Fresh full 42,643-entry traversal exceeded 240 s. The old completed full report
remains historical, not fresh evidence; no full-tree/all-byte/system-wide claim.
The partial H: venv remains unexecuted/unrepaired; original app processes were not
stopped. All new verification artifacts and documentation are inside NEWAPP.

**WARNING:** CPU default budget cost; browser click/download manual pending; fresh
full-inventory completion unavailable. **SUGGESTION:** return the confirmed ordinary
local-loader guarantee gap to writer; review explicit small CPU preset. Human labels
are required before precision/nDCG or any relevance-improvement claim.
**Next recommended: no new SDD phase.** Targeted writer correction/reverification,
then manual browser checks and human relevance assessment.

**Skill resolution: paths-injected** — read all three exact requested SKILL.md
paths again, shared verify protocol, own-environment Streamlit 1.64 references, and
Context7 Transformers offline-mode documentation. Original app logic was not edited.

---

## Historical verification report — superseded format policy, retained for audit

The section below is the pre-correction report, **not current status**. Safetensors
is required only for the reranker; official pinned BGE-M3 bin is now provisioned
and verified above. Its earlier model-format blocker is resolved. An untouched
copy is also in `.cache/evaluation/before-resume/real-verification-report.md`.

## Verification Report

**Change:** independent `app_reranker` against approved plan
`C:\Users\Usuario\.local\share\opencode\plans\1790797894034-proud-knight.md`
and `verification-status.json`. **Verdict: FAIL for complete-plan readiness;
partial verification achieved.** This is a contract incompatibility, not a proven
production-logic defect. No original source/environment/configuration changes,
production logic edits, installs, model conversions/replacements, delegation,
commits, or PRs were performed.

**Mode:** standard verification; `artifact_store.mode=none`; no new SDD pipeline.
No strict-TDD capability claim. Formal proposal/spec/design/task artifacts are not
supplied; formal spec compliance/design coherence and checked-task counts are
therefore skipped. The approved plan and concrete runtime requests below are the
acceptance contract. This operational report was explicitly requested under NEWAPP.

## Completeness

| Requested work | Evidence/status |
|---|---|
| Both pinned models, safetensors/no remote code | **PARTIAL/BLOCKED:** reranker provisioned; canonical pinned BGE-M3 has no published root safetensors |
| N=25/50/100/200 dual-model benchmark + separate offline benchmark | **UNTESTED/BLOCKED:** exact helper commands not run; they would load non-safetensors BGE-M3 |
| Electoral corruption + second concept aligned cosine/reranker top-ten | **UNTESTED/BLOCKED:** no carrier embeddings; no synthetic substitutes or fabricated rankings |
| Actual own launcher/server + real UI | **PARTIAL:** real-corpus AppTest shell and actual HTTP server passed; full real-model flow blocked |
| Documentation/status/evidence and final original inventory | **COMPLETE:** new files updated; fresh final inventory unchanged |
| Human relevance and browser row click/download | **MANUAL PENDING:** not mislabeled as technical runtime blockers or passed tests |

No formal task checklist was invented. Historical implementation tests remain
recorded independently from this batch's execution evidence.

## Blocking evidence and safe boundary

The read-only canonical helper declares BGE-M3 revision
`5617a9f61b028005a4858fdac845db406aefb181`. Its `load_model` at lines 187–210
explicitly checks Torch>=2.6 for official `pytorch_model.bin`, then uses
`AutoModel.from_pretrained` without `use_safetensors=True`. The public pinned
repository API and the read-only shared global snapshot both contain the bin,
not safetensors. No global-cache write/copy, BGE weight download/load, or canonical
helper modification was attempted.

Pinned API evidence:

- https://huggingface.co/api/models/BAAI/bge-m3/revision/5617a9f61b028005a4858fdac845db406aefb181
- https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3/revision/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e

Both API SHA values and published file lists are persisted in
`.cache/evaluation/provisioning.json`. Reranker weights were downloaded only to
NEWAPP using `hf_hub_download` with the exact SHA/cache directory. Hugging Face
Hub's built-in regular HTTP fallback was used because optional `hf_xet` is absent;
no dependency installation/workaround was performed. Available H: space before
provisioning was 149,337,948,160 bytes, and actual download succeeded.

Reranker snapshot:

`H:\Mi unidad\Silvana\discursos presidenciales\app nueva\app_reranker\.cache\huggingface\hub\models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`

Six required files were downloaded: config, tokenizer config/JSON, special-token
map, SentencePiece model, safetensors. Each size and SHA256 is recorded; LFS hashes
match published metadata. Weight size **2,271,071,852 bytes**; SHA256
`d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`.
The actual loader enforces `trust_remote_code=False`, `use_safetensors=True`, and
the exact pinned revision. Real model config commit, FP32 dtype and eval mode were
asserted during offline execution.

## Executed commands, tests and coverage

All commands ran in NEWAPP using the exact authorized Python:

`C:\Users\Usuario\AppData\Local\app-reranker\.venv\Scripts\python.exe`

`-B` / `PYTHONDONTWRITEBYTECODE=1` was used; canonical parent imports therefore
did not write external bytecode. NEWAPP runtime bootstrap selected H: HF/Torch/MPL/
TEMP caches. No original environment was executed or repaired.

| Command suffix after own Python | Runtime result |
|---|---|
| `-B C:\Users\Usuario\.agents\skills\developing-with-streamlit\scripts\discover.py --project-dir C:\Users\Usuario\AppData\Local\app-reranker` | Own Streamlit 1.64 installed skill located; testing/performance/CLI references read |
| `-I -B -m pip --isolated --disable-pip-version-check check` | No broken requirements found |
| `-B verify_real_runtime.py provision` | Partial, 77.5811 s, bounded 900,000 ms; reranker provisioned; BGE format blocker retained |
| `-B verify_real_runtime.py offline-smoke` | Passed, real Spanish passages and real safetensors; bounded 900,000 ms |
| `-B verify_real_runtime.py ui-shell` | Passed, real-corpus AppTest rendering/control checks; bounded 240,000 ms |
| `-B verify_real_runtime.py server` | Passed, actual own launcher/HTTP health/root, controlled stop; bounded 240,000 ms |
| `-B -m unittest -v test_verify_real_runtime` | **3 passed**, first 5.099 s / final 4.145 s, zero errors/failures; mocked download/revision/path guards only |
| `-I -B verify_sources.py` | 21 Python files parsed/compiled; original-app imports absent |
| `-B launch.py --check` | Own prefix/interpreter, user site false, CPU Torch, H: runtime paths confirmed |
| `-I -B inventory.py after` | First 240,000 ms attempt timed out; no remaining inventory process found; second bounded 900,000 ms attempt completed unchanged |
| `-B verify_real_runtime.py check-evidence` | Passed: final JSON/status/docs and completed evidence agree; baseline bytes hash verified; no repeat model inference or protected-tree traversal |

**Historical tests:** 66 mocked/isolation/UI + one real corpus integrity test passed
in the previous batch. Those 67 results were preserved, not rerun or claimed as
fresh independent results. The fresh smoke reloads and validates the real corpus
with its checksum/shape/norm checks and verifies read-only NPY mapping. **Coverage:**
no new percentage or threshold measured. **Build:** syntax/import-boundary checks
passed; there is no separate project build step in the supplied contract.

## Behavioral compliance matrix

This maps the supplied operational contract, not nonexistent formal delta specs.

| Behavior | Actual covering execution | Result |
|---|---|---|
| Pinned safe reranker loads locally | `offline-smoke`: model config SHA, eval/FP32, six-file cache provenance | COMPLIANT |
| Four carrier scores per real passage are finite | `(25,4)` shape/all-finite assertions and complete 100-pair metrics | COMPLIANT for smoke |
| Actual inference without networking | Local-only loader + both offline env flags + rejecting HTTP request and socket connect/connect_ex observers; zero attempts | COMPLIANT for reranker |
| Both query/reranker models resident with CPU metrics | Not executed | UNTESTED/BLOCKED |
| Candidate N budgets and real cosine comparison | Not executed; sampled passages explicitly are not cosine candidates | UNTESTED/BLOCKED |
| No implicit model loads from UI controls | Real-corpus AppTest startup, enable, country-filter, page-size; rejecting loader observers called zero times | COMPLIANT for shell |
| Actual server launch | Own launcher free 8502; `/_stcore/health` 200 `ok`, root 200 | COMPLIANT for HTTP startup |
| Analysis → Apply → paginate → pending filter → reapply → disable → exports with real weights | Not executed | UNTESTED/BLOCKED |
| Browser row click/download | No browser tools/Playwright/Selenium available in own environment; no installation attempted | MANUAL PENDING |
| Protected original inventory remains unchanged | Fresh completed `inventory.py after`, original baseline hash | COMPLIANT within declared scope |

## Actual attained metrics: reranker-only smoke

| Metric | Actual value |
|---|---|
| Selection | First 25 real corpus passages matching `electoral\|fraude\|voto`; **not cosine retrieval** |
| Carriers | Four default electoral-corruption formulations; full text saved |
| Strict local cold process load | 5.3560511 s, already provisioned cache; download excluded |
| Inference | 127.9023753 s |
| Pairs / truncations | 100 / 0; all four scores per passage preserved |
| CPU settings | Four Torch intra-op threads, batch 4, FP32, max 512 including specials |
| Sampled RSS peak | 2,347,937,792 bytes (2.35 GB / approximately 2.19 GiB), every 50 ms |
| OS peak working set | 2,348,015,616 bytes |
| Embeddings model resident | **No** |
| Offline network attempts | 0 |
| Precision@10 / nDCG@10 | **not measured** / **not measured** |

Full passage text, metadata, embedding-row identities, four logits, averages and
metrics are saved in `.cache/evaluation/reranker-offline-smoke.json`. These are
inference audit records, **not** requested aligned ranking comparisons.
The attained latency is a CPU warning for interactive use, but cannot establish
budget-200 latency, dual-model memory, production default suitability or relevance.
No N=1000 recommendation, linear extrapolation, or model-score quality claim is made.

## UI and server status at return

AppTest rendered the real corpus: **87,546 chunks, 297 documents, 1854–2024**.
The shell had no analysis query; explicit analysis was not clicked to avoid loading
the disallowed BGE weight format. Toggle/filter/page-size reruns loaded neither
model. Thus this is not a full AppTest real-model workflow or browser verification.

Own launcher chose free **8502**. Health: **200 `ok`**. Root: **200**, 7,260 bytes.
Created process tree: launcher **17828**, launcher base-Python wrapper **13780**,
Streamlit venv wrapper **18296**, Streamlit base-Python child **19004**. These
command lines are recorded in `.cache/evaluation/server-check.json`. Only verified
descendants created for this check were terminated, followed by the created
launcher. Waits confirmed they stopped. **No verification server is left running.**
No original or unrelated PID was stopped. HTTP success proves startup, not working
carrier analysis; no duplicate model resources were left resident.

## Original invariance and preserved evidence

Fresh final inventory: **42,643 size/mtime entries, 27 SHA256 entries, zero changes**.
Baseline `inventory-before.json` remains unchanged with SHA256:

`9c79e1c9bacf6e563536f95f3dbb953f18b468f849be6180f12bb0330c519236`

Scope: parent tree excluding NEWAPP; selected source/configuration hashes, old
environment metadata, all covered file/directory sizes and mtimes. This is **no
changes detected by inventory**, not an all-byte, global-cache, or system-wide proof.
The partial H: environment remains incident evidence, unexecuted/unrepaired.
The inventory timeout was retained as an execution limit, not an environment
installation incident; no environment workaround or replacement tool was used.

## Issues, verdict and next step

**CRITICAL:** the current request for both pinned models in safetensors conflicts
with the fixed canonical BGE-M3 repository and loader contract. Full dual-model
benchmark, two-concept comparison, offline carrier inference and real-model UI flow
remain blocked/unverified. This prevents complete-plan PASS.

**WARNING:** real reranker-only CPU smoke took 127.90 s for 100 pairs; neither
larger budgets nor dual-model peak memory has been measured. Browser selection and
actual download remain manual pending. Human relevance labels are absent; their
absence is not an installation/runtime defect.

**SUGGESTION / next recommended:** owner/writer should explicitly resolve whether
the canonical revision's official bin is permitted under the already-existing
Torch>=2.6 loader contract, retaining safetensors-only/no remote code for the
reranker. Do not silently substitute a revision, convert weights, or modify the
parent canonical helper. After resolution, finish the original benchmark and
strict offline benchmark, two-concept aligned top-ten audit, and full real UI flow.
Browser checks and human labeling follow. **No new SDD phase recommended.**

**Final verdict: FAIL for full completion; status: partial.** Reranker provisioning,
real offline inference, startup, helper tests and scoped original invariance passed.
No production logic findings were fixed by the verifier.

## Skill resolution

**paths-injected**, read before task work:

- `C:\Users\Usuario\.config\opencode\skills\sdd-verify\SKILL.md`
- `C:\Users\Usuario\.agents\skills\developing-with-streamlit\SKILL.md`
- `C:\Users\Usuario\.config\opencode\skills\cognitive-doc-design\SKILL.md`

Also read shared phase protocol/report template, the discovered own-environment
Streamlit 1.64 routing skill and `references/testing.md`, `performance.md`, `cli.md`.
Context7 Hub download/cache documentation was consulted for pinned/local-only
downloads; implementation uses the installed 0.36.2 API, not newer uninstalled APIs.
