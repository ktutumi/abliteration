"""verification-todo.md から docs/kanban.html を生成する.

MD の `### P...` 見出しを列に、`- [ ] **タイトル**` をカードに対応させる。
カード本文はタイトル行に続くインデント行。完了 `[x]` は「完了」列へ移す。

usage: python docs/build_kanban.py
"""

import html
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "verification-todo.md"
OUT = ROOT / "kanban.html"

TASK = re.compile(r"^- \[( |x)\] \*\*(.+?)\*\*\s*$")


def parse(md: str):
    columns: list[str] = []
    cards: list[dict] = []
    current = None
    in_code = False
    card = None

    for raw in md.splitlines():
        line = raw.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        if line.startswith("### "):
            current = line[4:].strip()
            columns.append(current)
            card = None
            continue
        if line.startswith("## "):
            current = None
            card = None
            continue

        m = TASK.match(line)
        if m:
            card = {"col": current, "title": m.group(2), "done": m.group(1) == "x", "body": []}
            cards.append(card)
            continue

        if card is not None and raw.startswith("  ") and line.strip():
            card["body"].append(line.strip())
        elif card is not None and not line.strip():
            continue
        else:
            card = None

    return [c for c in columns if any(k["col"] == c for k in cards)], cards


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def render(columns: list[str], cards: list[dict]) -> str:
    done = [c for c in cards if c["done"]]
    todo = [c for c in cards if not c["done"]]
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")

    def card_html(c: dict) -> str:
        tag = c["col"].split(":")[0] if c["col"] else ""
        body = "".join(f"<p>{inline(b)}</p>" for b in c["body"])
        cls = "card done" if c["done"] else "card"
        return (
            f'<article class="{cls}"><div class="tag">{html.escape(tag)}</div>'
            f'<h3>{inline(c["title"])}</h3>{body}</article>'
        )

    cols = []
    for col in columns:
        items = [c for c in todo if c["col"] == col]
        cols.append(
            f'<section><header><h2>{html.escape(col)}</h2>'
            f'<span class="count">{len(items)}</span></header>{"" .join(card_html(c) for c in items)}</section>'
        )
    cols.append(
        f'<section><header><h2>完了</h2><span class="count">{len(done)}</span></header>'
        f'{"" .join(card_html(c) for c in done)}</section>'
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Abliteration 検証 残課題</title>
<style>
  :root {{ --bg:#f6f7f9; --fg:#1f2328; --muted:#59636e; --line:#d8dee4; --card:#fff; --accent:#0969da; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
    font-family: system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif; line-height:1.6; }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .meta {{ color:var(--muted); font-size:12px; margin-bottom:20px; }}
  .meta code {{ background:#eef1f4; padding:1px 4px; border-radius:3px; }}
  .board {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px; align-items:start; }}
  section {{ background:#eceff3; border:1px solid var(--line); border-radius:10px; padding:10px; }}
  section > header {{ display:flex; align-items:baseline; gap:8px; padding:2px 4px 10px; }}
  h2 {{ font-size:13px; margin:0; letter-spacing:.02em; }}
  .count {{ margin-left:auto; font-size:12px; color:var(--muted); background:#dde3ea;
    border-radius:999px; padding:1px 8px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:10px; }}
  .card h3 {{ font-size:14px; margin:2px 0 6px; line-height:1.45; }}
  .card p {{ font-size:12.5px; color:#3d444d; margin:6px 0 0; }}
  .card p strong {{ color:var(--fg); }}
  .card.done {{ opacity:.62; }}
  .card.done h3 {{ text-decoration:line-through; }}
  .tag {{ font-size:10.5px; font-weight:700; color:var(--accent); letter-spacing:.04em; }}
  code {{ background:#eef1f4; padding:1px 4px; border-radius:3px; font-size:11.5px; }}
  @media (max-width: 700px) {{ body {{ padding:12px; }} }}
</style>
</head>
<body>
<h1>Abliteration 検証 残課題</h1>
<div class="meta">生成元: <code>docs/verification-todo.md</code> / 生成日時: {gen} / 未完了 {len(todo)} 件・完了 {len(done)} 件</div>
<div class="board">
{"".join(cols)}
</div>
</body>
</html>
"""


def main():
    columns, cards = parse(SRC.read_text(encoding="utf-8"))
    OUT.write_text(render(columns, cards), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT.parent)}: {len(columns)} 列 / {len(cards)} カード")


if __name__ == "__main__":
    main()
