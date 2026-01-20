"""
Complete Analysis Suite

Runs the full analysis including:
1. Main pipeline (volatility spike prediction)
2. Target comparison (direction vs volatility)
3. Stability check (performance by year)
4. Summary report
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 80)
print("COMPLETE ANALYSIS SUITE")
print("=" * 80)

print("\nThis will run:")
print("1. Main pipeline (volatility spike prediction)")
print("2. Target comparison (direction vs volatility spikes)")
print("3. Stability check (performance by year)")

response = input("\nProceed? (y/n): ")
if response.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

print("\n" + "=" * 80)
print("ANALYSIS 1: MAIN PIPELINE")
print("=" * 80)

os.system('python3 notebooks/run_pipeline_with_real_data.py')

print("\n" + "=" * 80)
print("ANALYSIS 2: TARGET COMPARISON")
print("=" * 80)

os.system('python3 notebooks/compare_targets.py')

print("\n" + "=" * 80)
print("ANALYSIS 3: STABILITY CHECK")
print("=" * 80)

os.system('python3 notebooks/stability_check.py')

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nSee ERROR_ANALYSIS.md for detailed error analysis and limitations.")
