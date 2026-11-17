"""
Script 11: Generate Publication-Grade Figures and Charts.
Reads experimental results, update logs, and predictions, producing vector-ready
figures for M.Tech Thesis and Scopus publication submission.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualization.report_plots import plot_method_comparison_bar, plot_event_metrics_comparison
from src.visualization.adaptation_plots import plot_adaptation_dynamics


def main():
    print("=" * 80)
    print("GENERATING PUBLICATION FIGURES AND EXPERIMENTAL CHARTS")
    print("=" * 80)

    fig_dir = os.path.join(PROJECT_ROOT, "experiments", "figures")
    reports_dir = os.path.join(PROJECT_ROOT, "experiments", "reports")
    update_dir = os.path.join(PROJECT_ROOT, "experiments", "update_logs")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Figure: Method Comparison Bar
    win_csv = os.path.join(reports_dir, "final_window_metrics_comparison.csv")
    if os.path.exists(win_csv):
        df_win = pd.read_csv(win_csv)
        fig_path = os.path.join(fig_dir, "fig3_window_metrics_comparison.png")
        plot_method_comparison_bar(df_win, metrics=["auroc", "auprc", "f1_score", "balanced_accuracy"], save_path=fig_path)
        print(f"[+] Generated: {fig_path}")

    # 2. Figure: Event-Based Clinical Metrics
    evt_csv = os.path.join(reports_dir, "final_event_metrics_comparison.csv")
    if os.path.exists(evt_csv):
        df_evt = pd.read_csv(evt_csv)
        fig_path = os.path.join(fig_dir, "fig4_clinical_event_metrics.png")
        plot_event_metrics_comparison(df_evt, save_path=fig_path)
        print(f"[+] Generated: {fig_path}")

    # 3. Figure: Adaptation Dynamics & Automated Rollbacks
    log_csv = os.path.join(update_dir, "e8_proposed_full_update_log.csv")
    if os.path.exists(log_csv):
        df_log = pd.read_csv(log_csv)
        logs = df_log.to_dict('records')
        fig_path = os.path.join(fig_dir, "fig5_online_adaptation_dynamics.png")
        plot_adaptation_dynamics(logs, title="Proposed Reliability-Gated TTA: Chronological Dynamics & Rollbacks", save_path=fig_path)
        print(f"[+] Generated: {fig_path}")

    # 4. Figure: Counterfactual Harm Prevention Pie & Bar
    cf_json = os.path.join(reports_dir, "counterfactual_analysis_results.json")
    if os.path.exists(cf_json):
        with open(cf_json, 'r', encoding='utf-8') as f:
            cf_data = json.load(f)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        
        # Gate Accuracy
        labels = ['Harmful Updates Blocked\n(True Negative Gating)', 'Harmful Updates Leaked\n(False Positive Gating)']
        sizes = [cf_data['harmful_correctly_rejected'], cf_data['harmful_accepted']]
        colors = ['#2ca02c', '#d62728']
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.08, 0))
        ax1.set_title(f"Harmful Update Prevention\n(Rejection Accuracy: {cf_data['harmful_rejection_accuracy']:.1f}%)", fontweight='bold')

        # Gate Precision
        ax2.bar(
            ['Harmful Rejection Accuracy', 'Reliability Gate Precision'],
            [cf_data['harmful_rejection_accuracy'], cf_data['gate_precision']],
            color=['#1f77b4', '#2ca02c'], width=0.45
        )
        ax2.set_ylabel('Percentage (%)', fontweight='bold')
        ax2.set_ylim([0, 115])
        ax2.set_title('Gate Reliability Metrics', fontweight='bold')
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        for p in ax2.patches:
            ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

        plt.tight_layout()
        cf_fig_path = os.path.join(fig_dir, "fig6_counterfactual_harm_rejection.png")
        plt.savefig(cf_fig_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] Generated: {cf_fig_path}")

    print("\nAll publication figures successfully created in:", fig_dir)


if __name__ == '__main__':
    main()
