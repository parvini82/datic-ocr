"""Output Formatting and Schema Enforcement for Phase 4."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
from ocr_solver.models import DetailedSolverResult, QuestionOutput


def format_single_result(result: Union[DetailedSolverResult, QuestionOutput]) -> Dict[str, Any]:
    """
    Format a solver result into the exact strict JSON dictionary specified in Phase 4:
    {
      "answer": "...",
      "question_text": "...",
      "changed": true/false,
      "original_ocr_text": "..."
    }
    """
    if isinstance(result, DetailedSolverResult):
        output = result.output
    else:
        output = result

    return {
        "answer": str(output.answer),
        "question_text": str(output.question_text),
        "changed": bool(output.changed),
        "original_ocr_text": str(output.original_ocr_text),
    }


def to_json_string(
    data: Union[DetailedSolverResult, QuestionOutput, List[Union[DetailedSolverResult, QuestionOutput]]],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> str:
    """Serialize result or list of results to pretty JSON string with Persian unicode preserved."""
    if isinstance(data, list):
        formatted = [format_single_result(item) for item in data]
    else:
        formatted = format_single_result(data)

    return json.dumps(formatted, indent=indent, ensure_ascii=ensure_ascii)


def generate_html_report(
    results: List[Union[Dict[str, Any], DetailedSolverResult, QuestionOutput]],
    output_path: Union[str, Path],
) -> str:
    """
    Generate a beautifully styled RTL HTML report from question solving results.

    Args:
        results: List of result dictionaries or DetailedSolverResult objects.
        output_path: Destination file path for the HTML report.

    Returns:
        The generated HTML content as a string.
    """
    normalized_results: List[Dict[str, Any]] = []
    for idx, r in enumerate(results, start=1):
        if isinstance(r, (DetailedSolverResult, QuestionOutput)):
            item = format_single_result(r)
            if isinstance(r, DetailedSolverResult) and r.image_path:
                item["image_name"] = Path(r.image_path).name
            else:
                item["image_name"] = f"Question #{idx}"
        else:
            item = dict(r)
            if "image_name" not in item:
                if "image_path" in item:
                    item["image_name"] = Path(item["image_path"]).name
                else:
                    item["image_name"] = f"سوال شماره {idx}"
        normalized_results.append(item)

    total_count = len(normalized_results)
    changed_count = sum(1 for r in normalized_results if r.get("changed"))
    unchanged_count = total_count - changed_count

    cards_html = []
    for idx, r in enumerate(normalized_results, start=1):
        img_name = r.get("image_name", f"سوال {idx}")
        ans = str(r.get("answer", "-"))
        changed = bool(r.get("changed", False))
        orig_ocr = str(r.get("original_ocr_text", "")).strip()
        final_text = str(r.get("question_text", "")).strip()

        changed_badge = (
            '<span class="badge badge-warning">تغییر یافته (OCR Refined)</span>'
            if changed
            else '<span class="badge badge-success">بدون تغییر (Direct Match)</span>'
        )

        card = f"""
        <div class="card">
            <div class="card-header">
                <div class="card-title-group">
                    <span class="card-index">#{idx}</span>
                    <h3 class="card-title">{img_name}</h3>
                </div>
                <div class="card-badges">
                    {changed_badge}
                    <span class="badge badge-primary">پاسخ: گزینه {ans}</span>
                </div>
            </div>
            
            <div class="card-body">
                <div class="text-section">
                    <div class="section-label">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                        متن اولیه OCR (استخراج شده)
                    </div>
                    <div class="content-box orig-box">{orig_ocr}</div>
                </div>

                <div class="text-section">
                    <div class="section-label section-label-final">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
                        متن نهایی سوال (مورد استفاده در حل)
                    </div>
                    <div class="content-box final-box">{final_text}</div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)

    html_template = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>گزارش تحلیل و حل هوشمند OCR (OCR-Aware Solver Report)</title>
    <!-- Google Fonts: Vazirmatn -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    
    <!-- MathJax for rendering LaTeX math formulas -->
    <script>
    window.MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        :root {{
            --bg-color: #0f172a;
            --surface-color: #1e293b;
            --surface-border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #3b82f6;
            --accent-primary-hover: #2563eb;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-cyan: #06b6d4;
            --box-orig-bg: #090e17;
            --box-final-bg: #0f2438;
            --font-family: 'Vazirmatn', Tahoma, sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: var(--font-family);
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.8;
            padding: 2.5rem 1.5rem;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--surface-border);
        }}

        .header h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .header p {{
            color: var(--text-secondary);
            font-size: 1.05rem;
        }}

        /* Stats Bar */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}

        .stat-card {{
            background: var(--surface-color);
            border: 1px solid var(--surface-border);
            border-radius: 1rem;
            padding: 1.25rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}

        .stat-info h4 {{
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }}

        .stat-info .stat-value {{
            font-size: 1.75rem;
            font-weight: 800;
            color: var(--text-primary);
        }}

        .stat-badge {{
            width: 48px;
            height: 48px;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }}

        .stat-badge.blue {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; }}
        .stat-badge.green {{ background: rgba(16, 185, 129, 0.15); color: #34d399; }}
        .stat-badge.amber {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; }}

        /* Cards List */
        .cards-list {{
            display: flex;
            flex-direction: column;
            gap: 1.75rem;
        }}

        .card {{
            background: var(--surface-color);
            border: 1px solid var(--surface-border);
            border-radius: 1.25rem;
            overflow: hidden;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .card:hover {{
            border-color: #475569;
            transform: translateY(-2px);
        }}

        .card-header {{
            padding: 1.25rem 1.75rem;
            background: rgba(30, 41, 59, 0.8);
            border-bottom: 1px solid var(--surface-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .card-title-group {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .card-index {{
            background: #334155;
            color: #94a3b8;
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.2rem 0.6rem;
            border-radius: 0.5rem;
        }}

        .card-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #f1f5f9;
        }}

        .card-badges {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}

        .badge {{
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .badge-primary {{
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: #ffffff;
            font-size: 1rem;
            padding: 0.4rem 1.1rem;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3);
        }}

        .badge-success {{
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-warning {{
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .card-body {{
            padding: 1.75rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}

        .text-section {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .section-label {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .section-label-final {{
            color: #38bdf8;
        }}

        .content-box {{
            padding: 1.25rem;
            border-radius: 0.75rem;
            font-size: 1.05rem;
            line-height: 1.9;
            white-space: pre-wrap;
            direction: rtl;
            text-align: right;
            border: 1px solid var(--surface-border);
        }}

        .orig-box {{
            background: var(--box-orig-bg);
            color: #cbd5e1;
            font-size: 0.98rem;
        }}

        .final-box {{
            background: var(--box-final-bg);
            color: #f8fafc;
            border-color: #0284c7;
            font-weight: 500;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 3.5rem;
            padding-top: 1.5rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
            border-top: 1px solid var(--surface-border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>گزارش نتایج حل هوشمند سوالات (OCR-Aware Solver)</h1>
            <p>نتایج استخراج، خوداصلاحی خطاها و حل گزینه‌ای سوالات آزمون‌های تستی فارسی</p>
        </header>

        <section class="stats-grid">
            <div class="stat-card">
                <div class="stat-info">
                    <h4>تعداد کل سوالات</h4>
                    <div class="stat-value">{total_count}</div>
                </div>
                <div class="stat-badge blue">📝</div>
            </div>
            <div class="stat-card">
                <div class="stat-info">
                    <h4>تطابق مستقیم</h4>
                    <div class="stat-value">{unchanged_count}</div>
                </div>
                <div class="stat-badge green">✓</div>
            </div>
            <div class="stat-card">
                <div class="stat-info">
                    <h4>اصلاح شده با بینایی ماشین</h4>
                    <div class="stat-value">{changed_count}</div>
                </div>
                <div class="stat-badge amber">🔍</div>
            </div>
        </section>

        <main class="cards-list">
            {cards_joined}
        </main>

        <footer class="footer">
            <p>OCR-Aware Solver Pipeline • Persian Exam Solving Agent</p>
        </footer>
    </div>
</body>
</html>
"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_template, encoding="utf-8")
    return html_template

