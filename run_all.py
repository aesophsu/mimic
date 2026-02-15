#!/usr/bin/env python3
"""
一键运行全流程（除 SQL 部分）

执行顺序（Python 流程 01-13，SQL 01/08 单独执行）：
  I. MIMIC 阶段: 01 → 02 → 03(仅 MIMIC) → 04(可选) → 05 → 06 → 06c(默认) → [06b可选] → 07
  II. eICU 阶段: 08 → 03(含 eICU 列) → 09 → 10 → [10b可选]
  III. 解释阶段: 11 → 12 → 13

注：03 在 mimic-only 时于 02 后运行（MIMIC-only Table 1）；全流程/eicu-only 时于 08 后运行（完整 Table 1 含 eICU 外部验证列）。

前置条件：
  - SQL 01 已执行 → data/raw/mimic_raw_data.csv
  - SQL 08 已执行 → data/raw/eicu_raw_data.csv（用于 II、III 阶段）

用法：
  uv run python run_all.py              # 全流程，遇错即停
  uv run python run_all.py --skip-04   # 跳过可选审计
  uv run python run_all.py --with-xgb-pruning  # 开启 06b（附加 legacy：XGBoost 重要性筛选）
  uv run python run_all.py --without-shap-pruning # 关闭默认 06c（仅开发集内 SHAP+Bootstrap 变量筛选）
  uv run python run_all.py --mimic-only   # 仅 MIMIC 阶段 (01-07)
  uv run python run_all.py --eicu-only    # 仅 eICU+解释 (08-13)
"""
import argparse
import os
import subprocess
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# 步骤定义：(阶段名, 工作目录, 脚本, 描述)
_STEP_DEFS = {
    "01": ("scripts/preprocess", "01_mimic_cleaning.py", "MIMIC 清洗"),
    "02": ("scripts/preprocess", "02_mimic_standardization.py", "MIMIC 标准化"),
    "03": ("scripts/audit_eval", "03_table1_baseline.py", "Table 1 基线表"),
    "04": ("scripts/audit_eval", "04_mimic_stat_audit.py", "MIMIC 审计（可选）"),
    "05": ("scripts/modeling", "05_feature_selection_lasso.py", "LASSO 特征选择"),
    "06": ("scripts/modeling", "06_model_training_main.py", "模型训练"),
    "06b": ("scripts/modeling", "06b_xgb_internal_feature_pruning.py", "XGBoost 开发集内变量筛选（可选）"),
    "06c": ("scripts/modeling", "06c_shap_bootstrap_feature_pruning.py", "SHAP+Bootstrap 开发集内变量筛选（可选）"),
    "07": ("scripts/modeling", "07_optimal_cutoff_analysis.py", "最优切点分析"),
    "08": ("scripts/preprocess", "08_eicu_alignment_cleaning.py", "eICU 对齐清洗"),
    "09": ("scripts/audit_eval", "09_cross_cohort_audit.py", "跨队列漂移审计"),
    "10": ("scripts/audit_eval", "10_external_validation_perf.py", "外部验证效能"),
    "10b": ("scripts/audit_eval", "10b_external_validation_slim.py", "精简版外部验证（可选）"),
    "11": ("scripts/audit_eval", "11_model_interpretation_shap.py", "SHAP 解释"),
    "12": ("scripts/audit_eval", "12_clinical_calibration_dca.py", "DCA 校准"),
    "13": ("scripts/audit_eval", "13_nomogram_odds_ratio.py", "列线图与森林图"),
}


def _build_steps(step_ids: list[str]) -> list[tuple[str, str, str, str]]:
    """按指定顺序构建步骤列表"""
    return [(sid, *_STEP_DEFS[sid]) for sid in step_ids if sid in _STEP_DEFS]


def check_prereq(step_id: str) -> tuple[bool, str]:
    """检查前置文件是否存在"""
    mimic_raw = os.path.join(PROJECT_ROOT, "data/raw/mimic_raw_data.csv")
    eicu_raw = os.path.join(PROJECT_ROOT, "data/raw/eicu_raw_data.csv")
    mimic_scale = os.path.join(PROJECT_ROOT, "data/cleaned/mimic_raw_scale.csv")
    mimic_steps = {"01", "02", "03", "04", "05", "06", "06b", "06c", "07"}
    eicu_steps = {"08", "09", "10", "11", "12", "13"}
    if step_id in mimic_steps and not os.path.exists(mimic_raw):
        return False, f"缺少 {mimic_raw}，请先执行 Step 01 SQL"
    if step_id in eicu_steps and not os.path.exists(eicu_raw):
        return False, f"缺少 {eicu_raw}，请先执行 Step 08 SQL"
    # 03 需要 mimic_raw_scale（由 01 产出），eICU-only 时需先跑过 01
    if step_id == "03" and not os.path.exists(mimic_scale):
        return False, f"缺少 {mimic_scale}，请先运行 Step 01"
    return True, ""


def run_step(step_id: str, workdir: str, script: str, desc: str, skip_04: bool) -> bool:
    """执行单步，返回是否成功"""
    if step_id == "04" and skip_04:
        print(f"\n[跳过] Step {step_id}: {desc}")
        return True
    ok, msg = check_prereq(step_id)
    if not ok:
        print(f"\n[跳过] Step {step_id}: {msg}")
        return True  # 跳过而非失败
    abs_workdir = os.path.join(PROJECT_ROOT, workdir)
    abs_script = os.path.join(abs_workdir, script)
    if not os.path.exists(abs_script):
        print(f"\n[错误] Step {step_id}: 脚本不存在 {abs_script}")
        return False
    print(f"\n{'='*60}")
    print(f"Step {step_id}: {desc}")
    print(f"  {workdir} / {script}")
    print("="*60)
    ret = subprocess.run(
        [sys.executable, script],
        cwd=abs_workdir,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="一键运行全流程（除 SQL）")
    parser.add_argument("--skip-04", action="store_true", help="跳过可选审计 Step 04")
    parser.add_argument("--mimic-only", action="store_true", help="仅运行 MIMIC 阶段 (01-07)")
    parser.add_argument("--eicu-only", action="store_true", help="仅运行 eICU+解释 (08-13)")
    parser.add_argument("--with-xgb-pruning", action="store_true", help="启用可选 Step 06b（附加 legacy：XGBoost 重要性筛选）")
    parser.add_argument("--without-shap-pruning", action="store_true", help="关闭默认 Step 06c（开发集内 SHAP+Bootstrap 变量筛选）")
    parser.add_argument("--with-slim-external", action="store_true", help="启用可选 Step 10b（默认读取 06c SHAP 推荐 k 的精简版外部验证）")
    parser.add_argument("--continue-on-error", action="store_true", help="遇错继续执行后续步骤")
    args = parser.parse_args()

    # mimic-only: 03 在 02 后（MIMIC-only Table 1）；06c 默认纳入精简筛选
    MIMIC_STEPS = ["01", "02", "03", "04", "05", "06", "06c", "07"]
    # 全流程/eicu-only: 03 在 08 后（完整 Table 1 含 eICU 列）；06c 默认纳入
    FULL_STEPS = ["01", "02", "04", "05", "06", "06c", "07", "08", "03", "09", "10", "11", "12", "13"]
    EICU_STEPS = ["08", "03", "09", "10", "11", "12", "13"]

    if args.without_shap_pruning:
        # 显式关闭默认 06c
        for seq in (MIMIC_STEPS, FULL_STEPS):
            if "06c" in seq:
                seq.remove("06c")

    if args.with_xgb_pruning:
        # 06b 作为附加 legacy 分析，默认插在 06c 后；若 06c 被关闭则插在 06 后
        for seq in (MIMIC_STEPS, FULL_STEPS):
            anchor = "06c" if "06c" in seq else "06"
            if anchor in seq and "06b" not in seq:
                seq.insert(seq.index(anchor) + 1, "06b")

    if args.with_slim_external:
        # 10b 仅作为附加分析，插入在 10 与 11 之间，不影响主流程默认行为
        for seq in (FULL_STEPS, EICU_STEPS):
            if "10" in seq and "10b" not in seq:
                seq.insert(seq.index("10") + 1, "10b")

    if args.mimic_only:
        steps_to_run = _build_steps(MIMIC_STEPS)
    elif args.eicu_only:
        steps_to_run = _build_steps(EICU_STEPS)
    else:
        steps_to_run = _build_steps(FULL_STEPS)

    print("\n" + "="*60)
    print("重症AP预测模型：全流程运行（除 SQL）")
    print("="*60)
    print(f"将执行 {len(steps_to_run)} 步: {', '.join(s[0] for s in steps_to_run)}")

    failed = []
    for step_id, workdir, script, desc in steps_to_run:
        ok = run_step(step_id, workdir, script, desc, args.skip_04)
        if not ok:
            failed.append(step_id)
            if not args.continue_on_error:
                print(f"\n❌ Step {step_id} 失败，已停止。")
                print("  使用 --continue-on-error 可继续执行后续步骤。")
                sys.exit(1)

    print("\n" + "="*60)
    if failed:
        print(f"⚠️ 完成，但有 {len(failed)} 步失败: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("✅ 全流程执行完成！")
        sys.exit(0)


if __name__ == "__main__":
    main()
