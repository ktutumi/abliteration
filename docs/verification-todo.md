# Abliteration 検証メモと残課題

> 作成日時: 2026-10-25 18:02

`Orion-zhen/abliteration` を Qwen3-0.6B と Qwen3.5-9B で実行・検証した結果と、そこから残った課題をまとめる。検証で作った評価ハーネスは `eval_refusal.py`（生成本文の拒否判定）と `refusal_propensity.py`（定型句の対数尤度）で、成果物は `eval_out/` に置いている（git 管理外）。

## Kanban ボードとの同期

`docs/kanban.html` はこのファイルから生成する実行用ボード。**このファイルを変更したら、必ず同じ変更の中で再生成する**（HTML を直接編集しない）。

```bash
python docs/build_kanban.py
```

- `### P1〜P4` の見出しが列、`- [ ] **タイトル**` がカードになる。`[x]` は「完了」列へ移動する
- カード本文はタイトル行に続く2スペースインデント行。`完了条件:` も本文として表示される
- 課題を追加するときは、既存の列見出しの下に同じ形式で追記する
- 列を増やすときは `### P5: ...` のように見出しを足すだけでよい

## 1. 現状の到達点

日本語は HarmBench-JA 200件、無害は XSTest-JA `refusal_required=no` 231件。英語は AdvBench 150件と XSTest `safe` 150件。

| モデル | JA 有害 | JA 無害 | EN 有害 | EN 無害 |
|---|---|---|---|---|
| Qwen3.5-9B（base） | 43.5% | 2.2% | 99.3% | 12.7% |
| 納品物（`~/models/Qwen--Qwen3.5-9B`） | 46.5% | 2.2% | 99.3% | 12.0% |
| + `out_proj` 対応（scale 1.0） | 38.5% | 1.7% | 99.3% | 12.0% |
| + `out_proj` 対応（scale 2.0） | **19.5%** | 3.0% | 99.3% | **6.0%** |

- 納品物は `linear_attn.out_proj`（32層中24層）が未改変で、挙動は base と区別できなかった。`out_proj` を対象に加えると動き出す。
- 日本語では狙った拒否が減り、無害側の拒否は増えていない。**英語では有害プロンプトの拒否が1件も変わらない**（base が 99.3% で天井）。
- 0.6B では scale 2.0 で日本語出力が 34% 英語化し、拒否率が逆に増えた（モデル破壊）。9B では同じ設定でも崩壊しない。

## 2. 残課題（優先度順）

### P1: 結論に直結する

- [ ] **英語で拒否が外れない原因の特定**
  base が 99.3% 拒否で、scale 2.0 + `out_proj` でも 150件中0件しか変化しない。一方で英語の過剰拒否（XSTest `safe`）は 12.7% → 6.0% と半減しており、「狙った拒否だけ残る」状態。層別の可視化（拒否方向との内積、残差ストリームへの寄与）と、`self_attn` 層に限定した強めのスケールで切り分ける。
  完了条件: 英語の有害拒否率が有意に下がる設定を見つける、または「この方向では英語の拒否を外せない」ことを機序つきで示す。

- [ ] **拒否が消えた応答の有害性評価**
  現在の指標は拒否表現の有無だけで、内容が危険かを見ていない。`APTO-001--APTO-HarmBench-JA` の `response_ja` / `refusal_required` を参照にして LLM ジャッジ（または人手）で判定する。
  完了条件: outproj+scale2 の JA 有害 200件について、拒否なし応答のうち何件が実害ある内容かの数値。

- [ ] **能力劣化の測定**
  拒否率だけでは「アブリテレーション」と「モデル破壊」を区別できない。0.6B は scale 2.0 で破綻したが、9B は出力長・多様性が base と同等という間接指標しかない。短い指示追従タスクか perplexity で base と比較する。
  完了条件: outproj+scale2 と base の性能差が数値で示されている。

- [ ] **0.6B scale 2.0 の言語崩壊の機序**
  `W − α·rrᵀW` を α=2 で引くと拒否方向の成分が符号反転する、という仮説。α=1.5 の中間点と `method: simple` で同じ α を試し、重みの反転を直接観測する。
  完了条件: 仮説の可否。中間 scale に「拒否は減り、出力は壊れない」領域があるか。

- [ ] **scale 3.0 以上は未検証**
  0.6B の破壊例があるため保留中。9B では言語崩壊指標（日本語出力率・4-gram 多様性）を見ながら上げる。

### P2: 計測の妥当性

- [ ] **拒否判定を LLM ジャッジへ**
  `eval_refusal.py` の判定は定型句マッチ（`REFUSAL`, 先頭200字）。日本語の「できません」単体は拾わず、「提供できません」等のみ拾う設計。判定漏れ・誤検出の上限が不明。
  完了条件: 目視した50件程度のラベルとの一致率が分かっている。

- [ ] **前置き型拒否の扱いを決める**
  冒頭では拒否せず、前置きの後で拒否する応答がある（全アームで1〜2.5%）。今回は全文走査で別集計したが、主指標は冒頭判定のまま。
  完了条件: 「拒否なし」の定義を文書化し、レポートで一貫させる。

- [ ] **サンプルサイズと検出力**
  200件では ±5pp 程度の差しか検出できない。`out_proj` 単体の効果（43.5% → 38.5%, p=0.099）は境界線上で、判断には不足。

- [ ] **データセットの拡張**
  未使用: XSTest-JA の `yes`（178件）と `conditional`（41件）、`APTO-001--APTO-SorryBench-JA`、`LLM-Tuning-Safety--HEx-PHI`、`walledai--StrongREJECT`、`walledai--HarmBench` contextual/copyright。
  完了条件: 少なくとも1つの独立した有害セットで同じ結論が再現する。

- [ ] **生成条件の固定とばらつき**
  greedy・`max_new_tokens=128`・`enable_thinking=False` 固定。温度サンプリングでの再現性と、128トークンで打ち切られる長い拒否の影響が未確認。
  完了条件: 温度0.7で同一プロンプトを複数回引いた拒否率のばらつきが分かっている。

### P3: ツール側の実装課題

- [ ] **`target_modules` のハードコード**
  `utils/ablation.py:105`。アーキテクチャごとに対象が変わる（今回の `out_proj` がまさにこれ）。config の `ablation.target_modules` に出して、既定値を従来どおりにする。
  完了条件: config で対象を変えられ、既存 config の挙動が変わらない。

- [ ] **`linear_attn.in_proj_*` を対象にすべきか**
  linear attention の入力側射影（`in_proj_qkv/z/b/a`）は未対象。出力側だけ削るのが妥当という判断だが、根拠が文書化されていない。

- [ ] **`mtp.layers.0` の誤マッチ**
  `utils/ablation.py:100-101` は最初の `layers` の次を層番号とするため、Qwen3.5 の `mtp.layers.0.*` が「層0」として扱われ、層0向けの設定（現状は既定値）が適用される。層0は chat template の共通末尾のせいで拒否方向がゼロなので実害は小さいが、`layer_overrides` に 0 を書くと MTP ヘッドに効いてしまう。

- [ ] **`layer_overrides` の無効な例**
  `config.example.yaml` と `config.yaml` の `114: {scale: 191, source_layer: 514}` は 32層モデルでは発火しない。既定の例から外すか、コメントで「層数に合わせて変更」と明記する。

- [ ] **使われていない設定項目**
  `inference.device`（`abliterate.py` は `device_map="auto"` 固定）と `inference.max_length`（トークナイズ時に切り詰めていない）は読まれるが機能していない。`max_lengh` のタイポ吸収（`utils/config.py:72`）も併せて整理する。

- [ ] **`clip` が `clip < 1.0` のときだけ効く**
  `utils/model.py:67`。既定 1.0 では事実上無効で、`magnitude_clip` 自体も単純 clamp（参照実装は閾値超過時にスケールする方式）。仕様を決めて統一する。

- [ ] **norm-preserving の非正方行列**
  `utils/math_utils.py:80` は `W.T` の行単位でノルムを保存する。`o_proj` / `out_proj`（4096×4096）では問題ないが、`down_proj`（4096×12288）では行が入力側ニューロンに対応するため「出力ニューロンごとのノルム保存」になっていない。更新方向自体は `simple` と等価。
  完了条件: 非正方行列での意図が明文化されている、または実装を直す。

- [ ] **`sparsify_tensor` の `topk` に `top_k` を流用**
  `abliterate.py:108` は `k=config.ablation.top_k` を渡すが、`top_k` は「拒否方向を平均する層数」の意味。`sparsify_method: topk` を選ぶと意味が衝突する。独立したキーに分ける。

- [ ] **`"layers."` を含むキー前提**
  `modify_shard_weights` は `"layers."` を含むキーしか対象にしないため、GPT 系の `h.N` 命名のモデルでは何も起きない（`compare.py` は `h` も見ている）。警告を出すか、層の特定方法を設定可能にする。

- [ ] **`torch.load(weights_only=False)`**
  `utils/io.py:126`。測定ファイルは pickle 相当なので、信頼できるファイルのみ読み込む前提を README かコードコメントで明示する。

- [ ] **層0の拒否方向が常にゼロ（設計メモ）**
  `welford_gpu_batched_multilayer_float32` は最終トークン位置の hidden state を取る。chat template の末尾が全プロンプト共通（`<|im_start|>assistant\n`）なので、埋め込み層の出力は全件同一になり、`refuse_0` のノルムは 0 になる。上位層選択では無視されるため実害はないが、`plot.py` の品質値が NaN になる（0除算）。

### P4: リポジトリ運用

- [ ] **E2E テストと CI**
  現状は `tests/` の unittest 27件のみで、`abliterate.py` の通し実行は検証されていない。小型モデル1本で「測定 → 改変 → 保存 → 評価」を回す E2E を CI に載せる。
  完了条件: PR ごとに E2E が回り、`out_proj` のような対象漏れを検出できる。

- [ ] **成果物の保存方針**
  `eval_out/` と `measurements*.pt` は `.gitignore` 済み。測定ファイル（9B で1.6MB）は再利用価値があるので、保存場所と命名規則を決める。

- [ ] **fork と upstream の同期方針**
  `origin` は上流（Orion-zhen）、`fork` は ktutumi。上流の更新を取り込む手順と、こちらの修正を upstream に PR するかを決める（`out_proj` 対応は上流にも有用な可能性がある）。

- [ ] **README への知見追記**
  上流 README には「`linear_attn` 構成のモデルでは `out_proj` も対象にする必要がある」「英語では拒否が外れにくい」といった知見がない。fork 側の README に検証結果へのリンクを置く。

- [ ] **実験設定の置き場所**
  `config_scale2.yaml` / `config_outproj.yaml` / `config_outproj_scale2.yaml` をリポジトリ直下に置いている。`configs/` に移すか、実験ログとして `docs/` にまとめる。

## 3. 再現コマンド

```bash
# 日本語の拒否率（有害 200 / 無害 231）
python eval_refusal.py \
  --harmful /home/tutumi/datasets/APTO-001--APTO-HarmBench-JA/train.csv --harmful-col prompt_ja \
  --safe /home/tutumi/datasets/APTO-001--APTO-XSTest-JA/train.csv --safe-col prompt_ja --safe-where refusal_required no \
  --models base=Qwen/Qwen3.5-9B abl=/home/tutumi/models/Qwen--Qwen3.5-9B \
  --max-new-tokens 128 --out eval_out/ja_qwen35_9b.json

# トークン単位の拒否傾向
python refusal_propensity.py \
  --prompts /home/tutumi/datasets/APTO-001--APTO-HarmBench-JA/train.csv --col prompt_ja --lang ja \
  --models base=Qwen/Qwen3.5-9B abl=/home/tutumi/models/Qwen--Qwen3.5-9B \
  --out eval_out/ja_qwen35_9b_propensity.json

# 再アブリテレーション（測定結果を再利用するので約2分）
python abliterate.py config_outproj_scale2.yaml
```

所要時間の目安: JA 評価 5分/モデル（9B）、EN 評価 4分/モデル、propensity 2分/モデル、再アブリテレーション 2分。
