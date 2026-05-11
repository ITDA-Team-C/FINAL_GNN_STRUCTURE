import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime


def create_html_report(model_name, metrics_dict, output_path="outputs/report.html"):
    """학습 결과를 HTML 리포트로 생성"""

    valid_metrics = metrics_dict.get("valid_metrics", {})
    test_metrics = metrics_dict.get("test_metrics", {})
    best_threshold = metrics_dict.get("best_threshold", 0.5)

    # 메트릭 포맷
    def fmt_metric(val):
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return str(val)

    # 메트릭 테이블 생성
    valid_rows = "".join([
        f"<tr><td>{k}</td><td>{fmt_metric(v)}</td></tr>"
        for k, v in valid_metrics.items()
    ])

    test_rows = "".join([
        f"<tr><td>{k}</td><td>{fmt_metric(v)}</td></tr>"
        for k, v in test_metrics.items()
    ])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GNN Fraud Detection - {model_name.upper()}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                line-height: 1.6;
                min-height: 100vh;
                padding: 20px;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}

            header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 20px;
                text-align: center;
            }}

            header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}

            header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}

            .content {{
                padding: 40px;
            }}

            .model-info {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                border-left: 4px solid #667eea;
            }}

            .model-info h3 {{
                color: #667eea;
                margin-bottom: 10px;
            }}

            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }}

            .info-item {{
                background: white;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #e0e0e0;
            }}

            .info-item label {{
                color: #666;
                font-weight: 600;
                display: block;
                margin-bottom: 5px;
                font-size: 0.9em;
            }}

            .info-item value {{
                font-size: 1.3em;
                color: #333;
                font-weight: bold;
            }}

            .metrics-section {{
                margin-bottom: 40px;
            }}

            .metrics-section h2 {{
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
                font-size: 1.8em;
            }}

            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}

            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                transition: transform 0.3s ease;
            }}

            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            }}

            .metric-card .label {{
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
            }}

            .metric-card .value {{
                font-size: 2.2em;
                font-weight: bold;
                font-family: 'Courier New', monospace;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}

            thead {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}

            th {{
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}

            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #e0e0e0;
            }}

            tbody tr:hover {{
                background: #f8f9fa;
            }}

            tbody tr:last-child td {{
                border-bottom: none;
            }}

            .chart-container {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}

            .chart {{
                width: 100%;
                height: 400px;
            }}

            footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 1px solid #e0e0e0;
                font-size: 0.9em;
            }}

            .timestamp {{
                color: #999;
                font-size: 0.9em;
            }}

            .badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                margin-right: 10px;
            }}

            .badge-success {{
                background: #d4edda;
                color: #155724;
            }}

            .badge-info {{
                background: #d1ecf1;
                color: #0c5460;
            }}

            @media (max-width: 768px) {{
                header h1 {{
                    font-size: 1.8em;
                }}

                .content {{
                    padding: 20px;
                }}

                .metrics-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔍 조직적 어뷰징 네트워크 탐지</h1>
                <p>GNN 기반 사기 탐지 모델 성능 분석 리포트</p>
            </header>

            <div class="content">
                <!-- 모델 정보 -->
                <div class="model-info">
                    <h3>📊 모델 정보</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <label>모델명</label>
                            <value>{model_name.upper()}</value>
                        </div>
                        <div class="info-item">
                            <label>최적 Threshold</label>
                            <value>{best_threshold:.4f}</value>
                        </div>
                        <div class="info-item">
                            <label>생성 날짜</label>
                            <value>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</value>
                        </div>
                    </div>
                </div>

                <!-- Validation 메트릭 -->
                <div class="metrics-section">
                    <h2>✅ Validation Set 성능</h2>
                    <div class="metrics-grid">
                        {_create_metric_cards(valid_metrics)}
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>지표</th>
                                <th>값</th>
                            </tr>
                        </thead>
                        <tbody>
                            {valid_rows}
                        </tbody>
                    </table>
                </div>

                <!-- Test 메트릭 -->
                <div class="metrics-section">
                    <h2>🎯 Test Set 성능 (Best Threshold 적용)</h2>
                    <div class="metrics-grid">
                        {_create_metric_cards(test_metrics)}
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>지표</th>
                                <th>값</th>
                            </tr>
                        </thead>
                        <tbody>
                            {test_rows}
                        </tbody>
                    </table>
                </div>

                <!-- 차트 -->
                <div class="chart-container">
                    <h3 style="margin-bottom: 20px; color: #333;">📈 메트릭 비교</h3>
                    <div id="chart" class="chart"></div>
                </div>
            </div>

            <footer>
                <p>GNN-based Systematic Abusing Network Detection | ITDA Team C</p>
                <p class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 생성</p>
            </footer>
        </div>

        <script>
            // 메트릭 비교 차트
            var metrics_keys = Object.keys({json.dumps(test_metrics)});
            var metrics_values = Object.values({json.dumps(test_metrics)});

            var trace = {{
                x: metrics_keys,
                y: metrics_values,
                type: 'bar',
                marker: {{
                    color: 'rgba(102, 126, 234, 0.8)',
                    line: {{
                        color: 'rgba(102, 126, 234, 1)',
                        width: 2
                    }}
                }}
            }};

            var layout = {{
                title: 'Test Set 성능 메트릭',
                xaxis: {{ title: '메트릭' }},
                yaxis: {{ title: '값' }},
                plot_bgcolor: '#f8f9fa',
                paper_bgcolor: 'white',
                font: {{ family: 'Segoe UI, sans-serif' }}
            }};

            Plotly.newPlot('chart', [trace], layout, {{responsive: true}});
        </script>
    </body>
    </html>
    """

    # 파일 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def _create_metric_cards(metrics):
    """메트릭 카드 HTML 생성"""
    cards_html = ""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            cards_html += f"""
            <div class="metric-card">
                <div class="label">{key.upper()}</div>
                <div class="value">{value:.4f}</div>
            </div>
            """
    return cards_html
