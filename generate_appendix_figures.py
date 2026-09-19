from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "outputs" / "hybrid_v2_final_statistics"
OUTPUT_DIR = SOURCE_DIR / "appendix_figures_clean"

NAVY = "#193854"
BLUE = "#2F66E5"
TEAL = "#147F75"
ORANGE = "#E27B00"
PURPLE = "#7137E8"
GREEN = "#16823B"
RED = "#C83A34"
GREY = "#6B7280"
LIGHT_GREY = "#D9E1E8"
PALE = "#F7F9FB"
WHITE = "#FFFFFF"


def load_font(size, bold=False):
    names = ["arialbd.ttf", "calibrib.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONTS = {
    "small": load_font(25),
    "body": load_font(30),
    "body_bold": load_font(30, True),
    "label": load_font(34),
    "label_bold": load_font(34, True),
    "panel": load_font(38, True),
}


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), str(text), font=font)
    return box[2] - box[0], box[3] - box[1]


def centered_text(draw, xy, text, font, fill=NAVY):
    w, h = text_size(draw, text, font)
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def right_text(draw, xy, text, font, fill=NAVY):
    w, h = text_size(draw, text, font)
    draw.text((xy[0] - w, xy[1] - h / 2), text, font=font, fill=fill)


def wrapped_lines(draw, text, font, max_width):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def centered_wrapped_text(draw, center_x, top_y, text, font, max_width, fill=NAVY, gap=5):
    lines = wrapped_lines(draw, text, font, max_width)
    _, line_h = text_size(draw, "Ag", font)
    for i, line in enumerate(lines):
        w, _ = text_size(draw, line, font)
        draw.text((center_x - w / 2, top_y + i * (line_h + gap)), line, font=font, fill=fill)


def pretty_label(value):
    mapping = {
        "reach_to_shelf": "Reach to shelf",
        "retract_from_shelf": "Retract from shelf",
        "hand_in_shelf": "Hand in shelf",
        "inspect_product": "Inspect product",
        "inspect_shelf": "Inspect shelf",
        "pickup_candidate": "Pickup",
        "product_held": "Product held",
        "return_candidate": "Return",
        "comparison_candidate": "Comparison",
        "background": "Background",
        "uncertain": "Uncertain",
        "hands": "Hands / product",
        "shelf": "Shelf",
    }
    return mapping.get(str(value), str(value).replace("_", " ").strip().title())


def save_image(image, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / filename, dpi=(300, 300), optimize=True)


def draw_legend(draw, series, center_x, y):
    widths = []
    for name, _, _ in series:
        widths.append(34 + 15 + text_size(draw, name, FONTS["body"])[0] + 44)
    x = center_x - sum(widths) / 2
    for width, (name, _, colour) in zip(widths, series):
        draw.rounded_rectangle((x, y, x + 34, y + 34), radius=5, fill=colour)
        draw.text((x + 49, y - 1), name, font=FONTS["body"], fill=NAVY)
        x += width


def draw_y_axis(draw, plot, y_max, tick_count=5, percent=False):
    x0, y0, x1, y1 = plot
    for i in range(tick_count + 1):
        value = y_max * i / tick_count
        y = y1 - (y1 - y0) * i / tick_count
        draw.line((x0, y, x1, y), fill=LIGHT_GREY, width=2)
        label = f"{value:.0%}" if percent else f"{value:.0f}"
        right_text(draw, (x0 - 20, y), label, FONTS["small"], GREY)
    draw.line((x0, y0, x0, y1), fill=NAVY, width=3)
    draw.line((x0, y1, x1, y1), fill=NAVY, width=3)


def grouped_bar_image(categories, series, y_max, filename, percent=True, size=(2500, 1400),
                      margin_left=150, margin_right=70, margin_top=135, margin_bottom=270,
                      tick_count=5, value_labels=True):
    image = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(image)
    plot = (margin_left, margin_top, size[0] - margin_right, size[1] - margin_bottom)
    draw_y_axis(draw, plot, y_max, tick_count, percent)
    draw_legend(draw, series, (plot[0] + plot[2]) / 2, 45)
    x0, y0, x1, y1 = plot
    group_width = (x1 - x0) / len(categories)
    inner_width = group_width * 0.72
    gap = max(7, int(inner_width * 0.025))
    bar_width = (inner_width - gap * (len(series) - 1)) / len(series)
    for category_i, category in enumerate(categories):
        group_x = x0 + group_width * (category_i + 0.5)
        first_x = group_x - inner_width / 2
        for series_i, (_, values, colour) in enumerate(series):
            value = float(values[category_i])
            bar_x0 = first_x + series_i * (bar_width + gap)
            bar_x1 = bar_x0 + bar_width
            bar_y0 = y1 - max(0, value) / y_max * (y1 - y0)
            draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, y1), radius=6, fill=colour)
            if value_labels:
                label = f"{value:.1%}" if percent else f"{value:.0f}"
                centered_text(draw, ((bar_x0 + bar_x1) / 2, max(y0 + 18, bar_y0 - 26)), label,
                              FONTS["small"], NAVY)
        centered_wrapped_text(draw, group_x, y1 + 32, category, FONTS["small"], group_width * 0.9, GREY)
    save_image(image, filename)


def grouped_bar_panel(draw, rect, categories, series, y_max, percent=False, panel_label=None,
                      tick_count=5, value_labels=False):
    x0, y0, x1, y1 = rect
    if panel_label:
        centered_text(draw, ((x0 + x1) / 2, y0 + 24), panel_label, FONTS["panel"], NAVY)
    legend_y = y0 + 62
    draw_legend(draw, series, (x0 + x1) / 2, legend_y)
    plot = (x0 + 125, y0 + 135, x1 - 35, y1 - 190)
    draw_y_axis(draw, plot, y_max, tick_count, percent)
    px0, py0, px1, py1 = plot
    group_width = (px1 - px0) / len(categories)
    inner_width = group_width * 0.72
    gap = max(5, int(inner_width * 0.025))
    bar_width = (inner_width - gap * (len(series) - 1)) / len(series)
    for category_i, category in enumerate(categories):
        group_x = px0 + group_width * (category_i + 0.5)
        first_x = group_x - inner_width / 2
        for series_i, (_, values, colour) in enumerate(series):
            value = float(values[category_i])
            bx0 = first_x + series_i * (bar_width + gap)
            bx1 = bx0 + bar_width
            by0 = py1 - max(0, value) / y_max * (py1 - py0)
            draw.rounded_rectangle((bx0, by0, bx1, py1), radius=4, fill=colour)
            if value_labels:
                label = f"{value:.1%}" if percent else f"{value:.0f}"
                centered_text(draw, ((bx0 + bx1) / 2, max(py0 + 15, by0 - 22)), label,
                              FONTS["small"], NAVY)
        centered_wrapped_text(draw, group_x, py1 + 25, category, FONTS["small"], group_width * 0.92, GREY)


def heat_colour(value):
    stops = [
        (0.0, (255, 253, 247)),
        (0.25, (252, 232, 170)),
        (0.55, (230, 155, 105)),
        (1.0, (190, 40, 42)),
    ]
    value = min(1.0, max(0.0, float(value)))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= value <= b:
            t = (value - a) / (b - a)
            return tuple(round(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return stops[-1][1]


def heat_text_colour(rgb):
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return WHITE if luminance < 0.58 else NAVY


def paste_rotated_text(image, text, font, center, fill=NAVY):
    probe = ImageDraw.Draw(image)
    width, height = text_size(probe, text, font)
    label = Image.new("RGBA", (width + 30, height + 30), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((15, 15), text, font=font, fill=fill)
    label = label.rotate(90, expand=True)
    image.paste(label, (int(center[0] - label.width / 2), int(center[1] - label.height / 2)), label)


def heatmap_image(rows, columns, values, filename, size, cell_font=None, label_font=None):
    image = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(image)
    cell_font = cell_font or FONTS["body"]
    label_font = label_font or FONTS["body_bold"]
    left = 430 if len(rows) <= 7 else 310
    top = 230 if len(columns) <= 7 else 180
    right = 80
    bottom = 150
    grid_x0, grid_y0 = left, top
    grid_x1, grid_y1 = size[0] - right, size[1] - bottom
    cell_w = (grid_x1 - grid_x0) / len(columns)
    cell_h = (grid_y1 - grid_y0) / len(rows)
    centered_text(draw, ((grid_x0 + grid_x1) / 2, 55), "Predicted", FONTS["panel"], NAVY)
    for j, column in enumerate(columns):
        centered_wrapped_text(draw, grid_x0 + (j + 0.5) * cell_w, 105, pretty_label(column),
                              label_font, cell_w * 0.95, NAVY, gap=0)
    for i, row in enumerate(rows):
        y = grid_y0 + (i + 0.5) * cell_h
        right_text(draw, (grid_x0 - 26, y), pretty_label(row), label_font, NAVY)
        for j, value in enumerate(values[i]):
            x0 = grid_x0 + j * cell_w
            y0 = grid_y0 + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            colour = heat_colour(value)
            draw.rectangle((x0, y0, x1, y1), fill=colour, outline=WHITE, width=3)
            if float(value) > 0:
                centered_text(draw, ((x0 + x1) / 2, (y0 + y1) / 2), f"{float(value):.1%}",
                              cell_font, heat_text_colour(colour))
    paste_rotated_text(image, "Ground truth", FONTS["panel"], (55, (grid_y0 + grid_y1) / 2))
    save_image(image, filename)


def short_metric(metric):
    replacements = {
        "Product origin top-candidate accuracy": "Exact product-origin accuracy",
        "Returned-boolean accuracy": "Returned-status accuracy",
        "MERL F1: reach_to_shelf": "MERL: Reach to shelf",
        "MERL F1: retract_from_shelf": "MERL: Retract from shelf",
        "MERL F1: hand_in_shelf": "MERL: Hand in shelf",
        "MERL F1: inspect_product": "MERL: Inspect product",
        "MERL F1: inspect_shelf": "MERL: Inspect shelf",
        "Product F1: pickup_candidate": "Product: Pickup",
        "Product F1: product_held": "Product: Product held",
        "Product F1: return_candidate": "Product: Return",
        "Product F1: comparison_candidate": "Product: Comparison",
    }
    return replacements.get(metric, metric)


def forest_image(data, filename, x_min, x_max, tick_step, change=False, size=(2600, 1850)):
    image = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 850, 120, 75, 155
    x0, x1 = left, size[0] - right
    y0, y1 = top, size[1] - bottom
    row_h = (y1 - y0) / len(data)

    def xpos(value):
        return x0 + (float(value) - x_min) / (x_max - x_min) * (x1 - x0)

    tick = math.ceil(x_min / tick_step) * tick_step
    while tick <= x_max + 1e-9:
        x = xpos(tick)
        draw.line((x, y0, x, y1), fill=LIGHT_GREY, width=2)
        display_tick = 0.0 if abs(tick) < 1e-9 else tick
        centered_text(draw, (x, y1 + 55), f"{display_tick:.0%}", FONTS["small"], GREY)
        tick += tick_step
    if x_min <= 0 <= x_max:
        x = xpos(0)
        draw.line((x, y0, x, y1), fill=NAVY, width=5)
    draw.line((x0, y1, x1, y1), fill=NAVY, width=3)
    for i, row in data.reset_index(drop=True).iterrows():
        y = y0 + (i + 0.5) * row_h
        label = short_metric(row["metric"])
        right_text(draw, (x0 - 35, y), label, FONTS["body"], NAVY)
        low, estimate, high = float(row["lower_95"]), float(row["estimate"]), float(row["upper_95"])
        stable = low > 0 or high < 0
        if change:
            colour = TEAL if low > 0 else RED if high < 0 else GREY
        else:
            colour = TEAL
        draw.line((xpos(low), y, xpos(high), y), fill=colour, width=8)
        draw.line((xpos(low), y - 14, xpos(low), y + 14), fill=colour, width=5)
        draw.line((xpos(high), y - 14, xpos(high), y + 14), fill=colour, width=5)
        r = 13 if stable or not change else 10
        draw.ellipse((xpos(estimate) - r, y - r, xpos(estimate) + r, y + r), fill=colour)
        label_x = min(x1 - 10, xpos(high) + 24)
        draw.text((label_x, y - 17), f"{estimate:+.1%}" if change else f"{estimate:.1%}",
                  font=FONTS["small"], fill=colour)
    save_image(image, filename)


def horizontal_bar_image(labels, values, filename, x_max, size=(2200, 2200), percent=True):
    image = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 250, 120, 70, 145
    x0, x1 = left, size[0] - right
    y0, y1 = top, size[1] - bottom
    row_h = (y1 - y0) / len(labels)
    for i in range(6):
        value = x_max * i / 5
        x = x0 + (x1 - x0) * value / x_max
        draw.line((x, y0, x, y1), fill=LIGHT_GREY, width=2)
        centered_text(draw, (x, y1 + 50), f"{value:.2f}" if percent else f"{value:.0f}",
                      FONTS["small"], GREY)
    draw.line((x0, y0, x0, y1), fill=NAVY, width=3)
    draw.line((x0, y1, x1, y1), fill=NAVY, width=3)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = y0 + i * row_h + row_h * 0.17
        h = row_h * 0.66
        right_text(draw, (x0 - 24, y + h / 2), str(label), FONTS["small"], NAVY)
        bx1 = x0 + (x1 - x0) * float(value) / x_max
        colour = TEAL if i else NAVY
        draw.rounded_rectangle((x0, y, bx1, y + h), radius=5, fill=colour)
        draw.text((min(x1 - 70, bx1 + 15), y + h / 2 - 14), f"{float(value):.2f}",
                  font=FONTS["small"], fill=NAVY)
    centered_text(draw, ((x0 + x1) / 2, size[1] - 42), "Priority score", FONTS["body"], NAVY)
    save_image(image, filename)


def attention_image(headline, confusion, filename):
    size = (2500, 1200)
    image = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(image)
    counts = np.asarray(confusion["counts"], dtype=float)
    total = counts.sum()
    overall = (counts[0, 0] + counts[1, 1]) / total
    metrics = [headline["attention_coverage"], headline["attention_accuracy_when_covered"], overall]
    categories = ["Coverage", "Accuracy when covered", "Overall agreement"]
    grouped_bar_panel(draw, (20, 20, 1280, 1170), categories,
                      [("Attention proxy", metrics, PURPLE)], 0.7, percent=True,
                      panel_label="Coverage and agreement", tick_count=7, value_labels=True)
    # Right-hand matrix.
    x0, y0, x1, y1 = 1500, 300, 2420, 850
    rows = confusion["row_labels"]
    cols = confusion["column_labels"]
    values = confusion["row_percentages"]
    cell_w = (x1 - x0) / len(cols)
    cell_h = (y1 - y0) / len(rows)
    centered_text(draw, ((x0 + x1) / 2, 100), "Expected and predicted target", FONTS["panel"], NAVY)
    centered_text(draw, ((x0 + x1) / 2, 170), "Predicted", FONTS["body_bold"], NAVY)
    for j, col in enumerate(cols):
        centered_text(draw, (x0 + (j + 0.5) * cell_w, 235), pretty_label(col), FONTS["body_bold"], NAVY)
    for i, row in enumerate(rows):
        right_text(draw, (x0 - 25, y0 + (i + 0.5) * cell_h), pretty_label(row), FONTS["body_bold"], NAVY)
        for j, value in enumerate(values[i]):
            bx0, by0 = x0 + j * cell_w, y0 + i * cell_h
            colour = heat_colour(value)
            draw.rectangle((bx0, by0, bx0 + cell_w, by0 + cell_h), fill=colour, outline=WHITE, width=3)
            centered_text(draw, (bx0 + cell_w / 2, by0 + cell_h / 2), f"{value:.1%}",
                          FONTS["label"], heat_text_colour(colour))
    save_image(image, filename)


def main():
    summary = pd.read_csv(SOURCE_DIR / "baseline_v2_headline_comparison.csv")
    merl = pd.read_csv(SOURCE_DIR / "merl_class_comparison.csv")
    product = pd.read_csv(SOURCE_DIR / "product_class_comparison.csv")
    duration = pd.read_csv(SOURCE_DIR / "duration_statistics.csv")
    paired = pd.read_csv(SOURCE_DIR / "paired_change_confidence_intervals.csv")
    bootstrap = pd.read_csv(SOURCE_DIR / "bootstrap_confidence_intervals.csv")
    priority = pd.read_csv(SOURCE_DIR / "per_video_priority_ranking.csv").sort_values("rank")
    statistics = json.loads((SOURCE_DIR / "statistics_data.json").read_text(encoding="utf-8"))

    # The six measures shown in the original headline comparison chart.
    headline_metrics = [
        "MERL mean class F1",
        "MERL mAP",
        "Exact multilabel sampled-frame accuracy",
        "Product definition-aligned mean F1",
        "Product strict tIoU mean F1",
        "Exact product-origin accuracy",
    ]
    headline = summary.set_index("metric").loc[headline_metrics]
    headline_labels = [
        "MERL mean class F1", "MERL mAP", "Exact frame accuracy",
        "Product mean F1", "Product strict-tIoU F1", "Exact origin accuracy",
    ]
    grouped_bar_image(headline_labels,
                      [("V1 baseline", headline["baseline"].tolist(), BLUE),
                       ("Hybrid V2", headline["hybrid_v2"].tolist(), TEAL)],
                      0.7, "appendix_01_v1_v2_summary.png")

    product_labels = [pretty_label(x) for x in product["label"]]
    grouped_bar_image(product_labels,
                      [("V1 baseline", product["baseline_f1"].tolist(), BLUE),
                       ("Hybrid V2", product["hybrid_v2_f1"].tolist(), TEAL)],
                      0.7, "appendix_02_product_v1_v2.png", size=(2300, 1250), margin_bottom=220)

    event_image = Image.new("RGB", (2600, 1250), WHITE)
    event_draw = ImageDraw.Draw(event_image)
    merl_labels = [pretty_label(x) for x in merl["label"]]
    grouped_bar_panel(event_draw, (20, 20, 1320, 1220), merl_labels,
                      [("Ground truth", merl["ground_truth_events"].tolist(), NAVY),
                       ("Predicted", merl["hybrid_v2_predicted_events"].tolist(), ORANGE)],
                      500, panel_label="MERL behaviours", value_labels=True)
    grouped_bar_panel(event_draw, (1320, 20, 2580, 1220), product_labels,
                      [("Ground truth", product["ground_truth_events"].tolist(), NAVY),
                       ("Predicted", product["hybrid_v2_predicted_events"].tolist(), ORANGE)],
                      500, panel_label="Product interactions", value_labels=True)
    event_draw.line((1300, 45, 1300, 1170), fill=LIGHT_GREY, width=3)
    save_image(event_image, "appendix_03_event_counts.png")

    grouped_bar_image(merl_labels,
                      [("Precision", merl["hybrid_v2_precision"].tolist(), NAVY),
                       ("Recall", merl["hybrid_v2_recall"].tolist(), BLUE),
                       ("F1", merl["hybrid_v2_f1"].tolist(), TEAL)],
                      1.0, "appendix_04_hybrid_v2_merl_performance.png", margin_bottom=235)

    grouped_bar_image(product_labels,
                      [("Precision", product["hybrid_v2_precision"].tolist(), NAVY),
                       ("Recall", product["hybrid_v2_recall"].tolist(), BLUE),
                       ("F1", product["hybrid_v2_f1"].tolist(), TEAL)],
                      0.8, "appendix_05_hybrid_v2_product_performance.png", size=(2300, 1250),
                      margin_bottom=220)

    base_merl = statistics["baseline_merl_confusion"]
    heatmap_image(base_merl["row_labels"], base_merl["column_labels"], base_merl["row_percentages"],
                  "appendix_06_merl_confusion_v1.png", (2600, 1500))
    v2_merl = statistics["merl_confusion"]
    heatmap_image(v2_merl["row_labels"], v2_merl["column_labels"], v2_merl["row_percentages"],
                  "appendix_07_merl_confusion_v2.png", (2600, 1500))

    base_origin = statistics["baseline_origin_shelf_confusion"]
    heatmap_image(base_origin["row_labels"], base_origin["column_labels"], base_origin["row_percentages"],
                  "appendix_08_origin_confusion_v1.png", (2600, 2250),
                  cell_font=load_font(26), label_font=load_font(28, True))
    v2_origin = statistics["origin_shelf_confusion"]
    heatmap_image(v2_origin["row_labels"], v2_origin["column_labels"], v2_origin["row_percentages"],
                  "appendix_09_origin_confusion_v2.png", (2700, 2350),
                  cell_font=load_font(25), label_font=load_font(27, True))

    paired_plot = paired.rename(columns={"absolute_change": "estimate"})
    forest_image(paired_plot, "appendix_10_paired_changes_95ci.png", -0.55, 0.75, 0.10, change=True)

    duration_image = Image.new("RGB", (2600, 1300), WHITE)
    duration_draw = ImageDraw.Draw(duration_image)
    merl_duration = duration[duration["domain"].eq("MERL ground truth")]
    product_duration = duration[duration["domain"].eq("Manual product ground truth")]
    duration_series_colours = ["#CFE0F5", BLUE, NAVY]
    grouped_bar_panel(duration_draw, (20, 20, 1320, 1270),
                      [pretty_label(x) for x in merl_duration["label"]],
                      [("Q1", merl_duration["q1_sec"].tolist(), duration_series_colours[0]),
                       ("Median", merl_duration["median_sec"].tolist(), duration_series_colours[1]),
                       ("Q3", merl_duration["q3_sec"].tolist(), duration_series_colours[2])],
                      7, panel_label="MERL event duration (seconds)", tick_count=7)
    grouped_bar_panel(duration_draw, (1320, 20, 2580, 1270),
                      [pretty_label(x) for x in product_duration["label"]],
                      [("Q1", product_duration["q1_sec"].tolist(), "#C8F5E9"),
                       ("Median", product_duration["median_sec"].tolist(), TEAL),
                       ("Q3", product_duration["q3_sec"].tolist(), GREEN)],
                      40, panel_label="Product-event duration (seconds)", tick_count=4)
    duration_draw.line((1300, 45, 1300, 1220), fill=LIGHT_GREY, width=3)
    save_image(duration_image, "appendix_11_event_durations.png")

    attention_image(statistics["headline"], statistics["attention_confusion"],
                    "appendix_12_attention_proxy.png")

    horizontal_bar_image(priority["video_id"].tolist(), priority["priority_score"].tolist(),
                         "appendix_13_per_video_priority_scores.png", 0.75)

    pickup = pd.DataFrame(statistics["pickup_origin_distribution"])
    grouped_bar_image(pickup["shelf"].tolist(),
                      [("Manual pickups", pickup["pickup_count"].tolist(), TEAL)],
                      45, "appendix_14_pickup_origin_distribution.png", percent=False,
                      size=(2400, 1100), margin_top=120, margin_bottom=170, tick_count=5)

    forest_image(bootstrap, "appendix_15_bootstrap_confidence_intervals.png",
                 0.0, 0.75, 0.10, change=False, size=(2500, 1050))

    expected = 15
    outputs = sorted(OUTPUT_DIR.glob("appendix_*.png"))
    if len(outputs) != expected:
        raise RuntimeError(f"Expected {expected} appendix figures, found {len(outputs)}")
    print(f"Created {len(outputs)} clean appendix figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
