"""
Build a PDF report of the ensemble-feasibility analysis chain for the Semarang
benchmark: error decorrelation (hard votes), masked re-analysis, soft-ensemble
evaluation on probability rasters, and consensus-error mapping.

Reads the CSV outputs already produced under the benchmark directory and the
figures, and writes ensemble_feasibility_report.pdf next to them.
"""

from pathlib import Path

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

BENCH = Path(
    "/Users/noel/Documents/Research_Projects/segwater_v2/outputs/evaluation/"
    "indonesia_inference_run_benchmark/"
    "semarang_probability_gt_0p5_overlap_benchmark__20260606T062610Z"
)
MASKED = BENCH / "ensemble_analysis" / "error_decorrelation_analysis_masked"
SOFT = BENCH / "ensemble_analysis" / "soft_ensemble_analysis"
OUT = BENCH / "ensemble_analysis" / "ensemble_feasibility_report.pdf"

MARGIN = 15
PAGE_W = 210 - 2 * MARGIN


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, "Ensemble feasibility analysis - Semarang benchmark",
                  align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, f"{self.page_no()}", align="C")
        self.set_text_color(0)

    def h1(self, text):
        self.set_font("helvetica", "B", 14)
        self.ln(3)
        self.multi_cell(PAGE_W, 7, text)
        self.ln(1)

    def h2(self, text):
        self.set_font("helvetica", "B", 11)
        self.ln(2)
        self.multi_cell(PAGE_W, 6, text)
        self.ln(1)

    def body(self, text):
        self.set_font("helvetica", "", 9.5)
        self.multi_cell(PAGE_W, 4.8, text)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("helvetica", "", 9.5)
        x = self.get_x()
        self.cell(5, 4.8, chr(149))
        self.multi_cell(PAGE_W - 5, 4.8, text)
        self.set_x(x)
        self.ln(0.8)

    def table(self, df, col_widths=None, font_size=8.5, header_fill=(230, 233, 240)):
        cols = list(df.columns)
        if col_widths is None:
            col_widths = [PAGE_W / len(cols)] * len(cols)
        self.set_font("helvetica", "B", font_size)
        self.set_fill_color(*header_fill)
        line_h = font_size * 0.55
        for c, w in zip(cols, col_widths):
            self.cell(w, line_h + 1.5, str(c), border=1, fill=True, align="C")
        self.ln()
        self.set_font("helvetica", "", font_size)
        for _, row in df.iterrows():
            for v, w in zip(row, col_widths):
                self.cell(w, line_h + 1.2, str(v), border=1, align="C")
            self.ln()
        self.ln(2.5)


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def shorten(name: str) -> str:
    return (name
            .replace("_native224_weighted", " (n224)")
            .replace("_large_crop_only_1024_b128_s1024", " (large)")
            .replace("Upernet_", "UperNet-")
            .replace("ConvnextV2base", "ConvNeXtV2")
            .replace("Swin_Base_224", "Swin-B")
            .replace("DPT_ViT_B_16", "DPT-ViT-B16")
            .replace("Segformer_MiT_B4", "SegFormer-MiT-B4")
            .replace("DeepLabV3plus_Resnet50", "DeepLabV3+-R50")
            .replace("UnetPlusPlus_Resnet50", "U-Net++-R50")
            .replace("Unet_Resnet50", "U-Net-R50"))


def main():
    pdf = Report()
    pdf.set_margins(MARGIN, MARGIN)
    pdf.set_auto_page_break(True, margin=16)

    # ------------------------------------------------------------ title page
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("helvetica", "B", 20)
    pdf.multi_cell(PAGE_W, 10,
                   "Is a Model Ensemble Worth It?\n"
                   "Error Decorrelation and Soft-Ensemble Analysis",
                   align="C")
    pdf.ln(4)
    pdf.set_font("helvetica", "", 12)
    pdf.multi_cell(PAGE_W, 7,
                   "Semarang benchmark - 6 S1 scenes with concurrent S2 reference\n"
                   "12 members: 7 architectures x 2 tiling strategies\n"
                   "GSHHG/GlobalSurfaceWater open-ocean mask applied",
                   align="C")
    pdf.ln(6)
    pdf.set_font("helvetica", "I", 10)
    pdf.multi_cell(PAGE_W, 6, "segwater_v2 - June 2026", align="C")

    pdf.ln(15)
    pdf.set_font("helvetica", "B", 11)
    pdf.multi_cell(PAGE_W, 6, "Key findings", align="C")
    pdf.set_font("helvetica", "", 9.5)
    for t in [
        "1. Model errors are NOT co-located: only ~10-23% of error pixels are shared by all members.",
        "2. The default threshold 0.5 is miscalibrated; optimal thresholds are 0.16-0.40.",
        "3. At threshold 0.5, soft ensembling improves 11 of 12 models - but not the best one (Swin-B).",
        "4. With tuned thresholds, only a cross-tiling pair beats the best single model (+0.34 IoU pts).",
        "5. Pixel-wise combiners capture <10% of the oracle headroom (0.71 achieved vs 0.90 reachable).",
        "6. The consensus-error core is 64% false positives - all 12 models contradict the reference there, pointing to reference noise.",
    ]:
        pdf.set_x(MARGIN + 8)
        pdf.multi_cell(PAGE_W - 16, 5.2, t)
        pdf.ln(0.5)

    # ------------------------------------------------- 1. setting / question
    pdf.add_page()
    pdf.h1("1. Question and data")
    pdf.body(
        "An ensemble can only fix a pixel if at least one member classifies it "
        "correctly: if all models fail on the same pixels, no combination rule helps. "
        "We test this precondition and then quantify achievable ensemble gains, on the "
        "6 Semarang evaluation scenes (S1 with concurrent S2-derived water reference, "
        "2144 x 2144 px, EPSG:4326, ~10 m). All metrics are pooled over scenes and "
        "computed on valid pixels only: reference nodata and the GSHHG/GlobalSurfaceWater "
        "open-ocean mask are excluded (~36% of each scene). Removing open ocean drops "
        "single-model IoU from ~0.91 to 0.51-0.71 - the coastal/inland zone is where the "
        "problem lives, and all numbers below refer to it.")
    pdf.body(
        "Members: DeepLabV3+, U-Net, U-Net++ (ResNet-50; 'CNN'), DPT-ViT-B16, "
        "SegFormer-MiT-B4, UperNet-Swin-B ('Transformer'), UperNet-ConvNeXtV2-base "
        "('Hybrid'); each inferred with two tiling strategies: native 224 weighted "
        "and large-crop-1024 (DPT and Swin-B exist only as native 224, since their "
        "backbones require 224 x 224 inputs). 12 members total. "
        "Errors are FP-or-FN pixels of the thresholded probability (comparison: "
        "greater-than).")

    # ------------------------------------------------- 2. decorrelation
    pdf.h1("2. Are the errors in the same places? (hard masks @0.5)")
    pdf.body(
        "Pairwise overlap of error masks (Jaccard) and the error-multiplicity "
        "distribution, masked, threshold 0.5:")
    t = pd.DataFrame({
        "group": ["native224 (7 models)", "large_crop (5 models)", "all 12 members"],
        "mean error Jaccard": ["0.549", "0.444", "0.500"],
        "wrong in ALL": ["22.8%", "18.0%", "11.7%"],
        "wrong in exactly 1": ["22.0%", "41.0%", "32.2%"],
        "oracle IoU": ["0.839", "0.833", "0.879"],
    })
    pdf.table(t, col_widths=[48, 40, 30, 34, 28])
    pdf.body(
        "Errors are substantially decorrelated: ~77-88% of error pixels are correct in "
        "at least one member, so the ensemble precondition holds, with a large oracle "
        "headroom (best single: 0.705). Mechanism structure: same-family CNNs share "
        "blind spots (CNN-CNN error Jaccard 0.64/0.48; U-Net vs U-Net++ up to 0.66, "
        "driven by shared false negatives), while CNN x Transformer pairs are the most "
        "independent (0.41-0.51). All models are FN-dominated at 0.5 "
        "(precision >> recall: missed water).")

    pdf.h2("2.1 Single models (pooled IoU, masked, threshold 0.5)")
    singles = pd.read_csv(SOFT / "soft_ensemble_results.csv")
    s1 = singles[singles.k == 1].sort_values("iou_05", ascending=False)
    t = pd.DataFrame({
        "member": [shorten(m) for m in s1["members"]],
        "IoU @0.5": [fmt(v) for v in s1["iou_05"]],
        "macro @0.5": [fmt(v) for v in s1["iou_05_macro"]],
        "IoU tuned": [fmt(v) for v in s1["iou_loo_tuned"]],
        "macro tuned": [fmt(v) for v in s1["iou_loo_tuned_macro"]],
        "thr": [fmt(v, 2) for v in s1["mean_tuned_threshold"]],
        "AUC-PR": [fmt(v) for v in s1["auc_pr"]],
    })
    pdf.table(t, col_widths=[58, 22, 22, 22, 23, 14, 19], font_size=8)
    pdf.body(
        "Micro = IoU of pooled pixel counts across the 6 scenes; macro = mean of "
        "per-scene IoUs (each scene weighted equally, as in the AUC-ROC evaluation "
        "pipeline). The two agree within ~0.01 for every member because the scenes "
        "are homogeneous in water fraction (11-16%) and difficulty; all rankings and "
        "conclusions in this report are identical under either aggregation.")

    # ------------------------------------------------- 3. hard votes
    pdf.h1("3. Hard majority voting (binary masks @0.5)")
    t = pd.DataFrame({
        "ensemble (majority vote, tie -> water)": [
            "native224: ALL-7", "native224: CNN-3", "native224: Transformer-3",
            "native224: best pair (ConvNeXtV2-n224 + Swin-B)",
            "large_crop: ALL-5", "large_crop: CNN-3",
            "large_crop: best pair (ConvNeXtV2 + U-Net)"],
        "IoU": ["0.6532", "0.5940", "0.6906", "0.7121",
                "0.6358", "0.5932", "0.6929"],
        "vs best single": ["-0.052", "-0.111", "-0.015", "+0.007",
                           "-0.031", "-0.074", "+0.026"],
    })
    pdf.table(t, col_widths=[105, 35, 40])
    pdf.body(
        "Exhaustive enumeration of all 127 + 31 subsets shows: voting many models is "
        "counterproductive (correlated CNN false negatives steer the majority), pure "
        "three-mechanism ensembles never win, and the best subsets are always pairs of "
        "one strong transformer/hybrid plus one decorrelated partner. A single good CNN "
        "partner does not hurt - what hurts is CNN redundancy (>=2 CNNs) and strict odd "
        "majorities in an FN-dominated regime (even k with tie->water outperforms "
        "adjacent odd k). The k=2 tie->water rule equals a union of water masks, "
        "trading precision for the recall these models lack.")

    # ------------------------------------------------- 4. soft ensembles
    pdf.h1("4. Soft ensembles on the probability rasters")
    pdf.body(
        "Unweighted means of member probabilities for 194 candidate sets (all "
        "within-tiling subsets, all 66 cross-member pairs, ALL-12), evaluated (a) at "
        "the default threshold 0.5 exactly, and (b) at a leave-one-scene-out tuned "
        "threshold (tune on 5 scenes, evaluate on the held-out one, pool the 6 "
        "held-out results). Sanity gate: each member at 0.5 reproduces the "
        "confusion-raster metrics (passed).")

    pdf.h2("4.1 The threshold is half the story")
    pdf.body(
        "Optimal thresholds are 0.16-0.40 for 11 of 12 members (table in section 2.1); "
        "the default 0.5 costs 0.5-7 IoU points per model. Tuning Swin-B alone lifts "
        "0.7052 -> 0.7114, recovering most of what hard voting achieved (0.7121). The "
        "exception is DeepLabV3+ large-crop (tunes to 0.65), consistent with its "
        "anomalous coastal false positives.")

    pdf.h2("4.2 Does soft ensembling help at the DEFAULT threshold 0.5?")
    pdf.body(
        "Not for the best model: no ensemble of any size beats Swin-B native224 alone "
        "(0.7052; best ensemble 0.7011). But every other member is improved at 0.5 by "
        "averaging with a stronger or decorrelated partner - often massively:")
    t = pd.DataFrame({
        "member": ["U-Net++-R50 (n224)", "DeepLabV3+-R50 (large)",
                   "UperNet-ConvNeXtV2 (n224)", "U-Net-R50 (n224)",
                   "DPT-ViT-B16 (n224)", "UperNet-Swin-B (n224)"],
        "single IoU @0.5": ["0.539", "0.513", "0.638", "0.609", "0.668", "0.705"],
        "best ensemble with it @0.5": ["0.675", "0.665", "0.692", "0.685",
                                       "0.699", "0.701"],
        "improved?": ["yes", "yes", "yes", "yes", "yes", "no"],
    })
    pdf.table(t, col_widths=[62, 38, 50, 30])

    pdf.h2("4.3 With tuning made fair")
    res = pd.read_csv(SOFT / "soft_ensemble_results.csv")
    top = res.sort_values("iou_loo_tuned", ascending=False).head(6)

    def compact(name: str) -> str:
        return (shorten(name).replace(",", " + ")
                .replace("UperNet-", "").replace("-MiT-B4", "")
                .replace("-ViT-B16", "").replace("-R50", ""))

    t = pd.DataFrame({
        "member set (soft mean)": [compact(m) for m in top["members"]],
        "tilings": top["tilings"].str.replace("large_crop_1024", "large")
                                  .str.replace("native224", "n224").values,
        "IoU @0.5": [fmt(v) for v in top["iou_05"]],
        "IoU tuned": [fmt(v) for v in top["iou_loo_tuned"]],
        "macro tuned": [fmt(v) for v in top["iou_loo_tuned_macro"]],
        "AUC-PR": [fmt(v) for v in top["auc_pr"]],
    })
    pdf.table(t, col_widths=[68, 28, 22, 22, 22, 18], font_size=8)
    pdf.body(
        "Only the cross-tiling pair ConvNeXtV2-large + Swin-B-n224 meaningfully beats "
        "tuned Swin-B (0.7148 vs 0.7114, +0.34 pts); it is also the only set whose "
        "AUC-PR exceeds Swin-B's (0.9189 vs 0.9173), i.e. the only genuinely better "
        "probability ranking. Within a single tiling strategy, no ensemble improves on "
        "tuned Swin-B (best same-tiling set: +0.0003, noise). Weighted averaging "
        "(0.712) and a per-pixel logistic-regression stacker (0.709) do not beat the "
        "plain mean - with 6 scenes there is nothing extra parameters can learn that "
        "survives cross-validation. Architectural diversity alone is therefore "
        "insufficient; the residual ensemble gain is attributable to the tiling/"
        "windowing axis.")

    # ------------------------------------------------- 5. why
    pdf.h1("5. Why pixel-wise combiners cannot cash in the headroom")
    pdf.body(
        "Re-measuring decorrelation at tuned thresholds rules out the obvious "
        "suspicion that tuning re-correlates the errors:")
    t = pd.DataFrame({
        "group": ["native224 (7)", "native224 (7)", "all 12", "all 12"],
        "thresholds": ["0.5", "tuned", "0.5", "tuned"],
        "mean error Jaccard": ["0.549", "0.547", "0.500", "0.483"],
        "shared by all": ["22.8%", "22.6%", "11.7%", "9.5%"],
        "oracle IoU": ["0.839", "0.848", "0.879", "0.903"],
    })
    pdf.table(t, col_widths=[40, 30, 40, 35, 35])
    pdf.body(
        "Errors stay equally decorrelated and the oracle even improves (0.903), yet "
        "realized ensembles reach only ~0.715. Two facts explain the gap. First, the "
        "complementary signal is weak per pixel: where tuned Swin-B errs, the best "
        "partner is correct on only 29% of pixels. Second, the decorrelated errors are "
        "mostly low-margin, near-threshold disagreements (mixed pixels, water edges, "
        "ambiguous SAR returns): which member errs there is quasi-random, and the "
        "averaged probability lands near the boundary too. Averaging cancels "
        "independent confident mistakes; these models barely make any that a partner "
        "confidently contradicts. The oracle exploits per-pixel hindsight, which no "
        "realizable pixel-wise combiner has.")

    # ------------------------------------------------- 6. consensus errors
    pdf.add_page()
    pdf.h1("6. Where is the irreducible core? Consensus-error mapping")
    pdf.body(
        "Per-scene GeoTIFFs (soft_ensemble_analysis/consensus_error_rasters/) encode, "
        "for every valid pixel, how many of the 12 members misclassify it at tuned "
        "thresholds and at 0.5, plus the reference class. Pixels wrong in >=10 members "
        "(75k-107k per scene) form coherent objects, not noise:")
    t = pd.DataFrame({
        "consensus errors (>=10 of 12 wrong, pooled)": [
            "false positives (models say water, reference says land)",
            "false negatives (missed water)"],
        "pixels": ["323,935", "185,272"],
        "share": ["64%", "36%"],
    })
    pdf.table(t, col_widths=[110, 35, 35])
    pdf.body(
        "The consensus core is FP-dominated - a reversal of the overall FN-dominated "
        "error profile. The consensus-FP blobs are large, compact, inland, and grow in "
        "the 2023/2024 scenes; when 12 independently trained architectures across two "
        "tiling strategies unanimously contradict the reference on a coherent patch, "
        "reference noise is the likely cause (flooded vegetation or wet soil visible "
        "to SAR but rejected by the NDWI/MNDWI/AWEI-veto reference, or genuine change "
        "between the S1 and S2 acquisitions). Consensus FNs instead concentrate in the "
        "aquaculture-pond strip: narrow dikes and sub-pixel ponds at 10 m resolution - "
        "the honest irreducible model-side errors. Part of the 'irreducible 22%' is "
        "therefore probably not model failure at all, implying every benchmark IoU is "
        "somewhat underestimated.")
    fig = SOFT / "consensus_error_rasters" / "consensus_errors_overview.png"
    if fig.exists():
        pdf.ln(1)
        pdf.image(str(fig), x=MARGIN, w=PAGE_W)
        pdf.set_font("helvetica", "I", 8.5)
        pdf.multi_cell(PAGE_W, 4.5,
                       "Consensus errors at tuned thresholds (12 members), one panel "
                       "per evaluation scene. Blue: >=10 members miss water (consensus "
                       "FN). Red: >=10 members predict water where the reference says "
                       "land (consensus FP). Light blue/orange: recoverable "
                       "minority/majority errors.")

    # ------------------------------------------------- 7. recommendations
    pdf.add_page()
    pdf.h1("7. Conclusions and recommendations")
    for b in [
        "Re-tune the operating threshold first (about 0.35 for the strong models). It is "
        "free and worth 0.5-7 IoU points per model; part of the apparent ensemble gain "
        "was threshold correction in disguise.",
        "If dual inference is affordable, average ConvNeXtV2-large-crop with "
        "Swin-B-native224 and threshold at about 0.37: IoU 0.7148, the only combination "
        "with a genuinely better probability ranking than the best single model.",
        "Do not deploy plain majority votes of many models, all-CNN ensembles, or "
        "within-tiling soft ensembles: none beats a threshold-tuned Swin-B.",
        "Skip weighted averaging and per-pixel stacking at this data scale; they cannot "
        "beat the unweighted mean with 6 reference scenes.",
        "The remaining headroom (0.715 -> 0.903 oracle) is real but requires combiners "
        "with spatial context (e.g. a small CNN fusing member probability maps, or "
        "per-region model selection) and, above all, more reference scenes.",
        "Audit the reference in the consensus-FP blobs (band 1 >= 10, band 3 = 0 in the "
        "consensus rasters) against S1 backscatter and concurrent S2 before further "
        "benchmarking: the irreducible error core is 64% 'all models contradict the "
        "reference', and adjudicating it may raise every reported IoU.",
    ]:
        pdf.bullet(b)

    pdf.h2("Artifacts")
    pdf.set_font("courier", "", 8)
    for line in [
        "scripts/evaluation/analyze_error_decorrelation.py",
        "scripts/evaluation/exhaustive_hard_vote_search.py",
        "scripts/evaluation/evaluate_soft_ensembles.py",
        "scripts/evaluation/generate_consensus_error_rasters.py",
        "<benchmark>/error_decorrelation_analysis{,_masked}/  (CSVs, heatmaps, REPORT.md)",
        "<benchmark>/soft_ensemble_analysis/                  (CSVs, REPORT.md)",
        "<benchmark>/soft_ensemble_analysis/consensus_error_rasters/*.tif",
    ]:
        pdf.cell(PAGE_W, 4.2, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
