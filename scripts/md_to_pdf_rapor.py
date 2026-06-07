"""Convert EK_RAPOR_TABLOLARI.md to 24_yazlab2_rapor.pdf."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT_DIR = Path(r"C:\Windows\Fonts")
INPUT_MD = Path(r"c:\Users\oguzhan\Desktop\EK_RAPOR_TABLOLARI.md")
OUTPUT_PDF = Path(r"c:\Users\oguzhan\Desktop\24_yazlab2_rapor.pdf")


def register_fonts() -> tuple[str, str]:
    regular = "Arial"
    bold = "Arial-Bold"
    pdfmetrics.registerFont(TTFont(regular, str(FONT_DIR / "arial.ttf")))
    pdfmetrics.registerFont(TTFont(bold, str(FONT_DIR / "arialbd.ttf")))
    return regular, bold


def make_styles(regular: str, bold: str):
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold,
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=regular,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=12,
            leading=15,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName=bold,
            fontSize=10.5,
            leading=13,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName=regular,
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "note": ParagraphStyle(
            "Note",
            parent=base["Normal"],
            fontName=regular,
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName=regular,
            fontSize=9,
            leading=11,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName=bold,
            fontSize=9,
            leading=11,
        ),
    }


def build_table(headers: list[str], rows: list[list[str]], col_widths=None):
    data = [
        [Paragraph(h, styles["table_header"]) for h in headers],
        *[[Paragraph(str(c), styles["table_cell"]) for c in row] for row in rows],
    ]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5496")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFBFBF")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F2F2F2")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


regular_font, bold_font = register_fonts()
styles = make_styles(regular_font, bold_font)

story = []

story.append(Paragraph("YazLab 2. Proje — Deney Sonuçları ve Karşılaştırmalı Analiz Tabloları", styles["title"]))
story.append(
    Paragraph(
        "<b>24. Grup</b> | İbrahim Alperen KESKİN (231307052), Oğuzhan ATILKAN (231307085)",
        styles["subtitle"],
    )
)
story.append(
    Paragraph(
        "<b>Deney ID:</b> 20260607_044225 | <b>Tarih:</b> 7 Haziran 2026",
        styles["subtitle"],
    )
)
story.append(Spacer(1, 0.2 * cm))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2F5496")))
story.append(Spacer(1, 0.3 * cm))

story.append(
    Paragraph(
        "Bu tamamlayıcı doküman, <i>From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis</i> "
        "projesinin raporunda yer alması gereken deney sonuçlarını içermektedir. Tüm sayılar tam deney logundan alınmıştır.",
        styles["body"],
    )
)
story.append(
    Paragraph(
        "<b>Kaynak dosyalar:</b> logs/exp_20260607_044225.json, logs/experiment_summary.csv, "
        "logs/runtime_summary.csv, logs/unseen_summary.csv",
        styles["note"],
    )
)

# Section 1
story.append(Paragraph("1. Temel Performans ve Stabilite", styles["h2"]))
story.append(
    Paragraph(
        "5 random seed (42, 123, 2026, 7, 999) ile elde edilen ortalama F1-skor ve standart sapma değerleri. "
        "SKAB: 5-fold GroupKFold ortalaması. BATADAL: kronolojik test kümesi.",
        styles["body"],
    )
)
story.append(Paragraph("Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)", styles["h3"]))
story.append(
    build_table(
        ["Model", "SKAB", "BATADAL"],
        [
            ["LSTM", "0.1716 ± 0.0260", "0.7642 ± 0.0043"],
            ["GRU", "0.1214 ± 0.0278", "0.7680 ± 0.0037"],
            ["1D-CNN", "0.2614 ± 0.0291", "0.7699 ± 0.0016"],
            ["Automata", "0.0311 ± 0.0000", "0.0202 ± 0.0000"],
        ],
        col_widths=[4 * cm, 5.5 * cm, 5.5 * cm],
    )
)

# Section 2
story.append(Paragraph("2. Gürültü ve Unseen Veri Analizi (Robustness)", styles["h2"]))
story.append(
    Paragraph(
        "Gaussian gürültü (σ=0.05) ve unseen pattern senaryosu. Det. Rate ve Map. Acc. yalnızca otomata için geçerlidir.",
        styles["body"],
    )
)
story.append(Paragraph("Tablo 2a: SKAB", styles["h3"]))
story.append(
    build_table(
        ["Model", "Orijinal (F1)", "Gürültülü (F1)", "Det. Rate", "Map. Acc."],
        [
            ["LSTM", "0.1716", "0.1672", "—", "—"],
            ["GRU", "0.1214", "0.1381", "—", "—"],
            ["1D-CNN", "0.2614", "0.3343", "—", "—"],
            ["Automata", "0.0311", "0.0465", "0.0034", "0.7692"],
        ],
        col_widths=[3.2 * cm, 3.2 * cm, 3.2 * cm, 2.8 * cm, 2.8 * cm],
    )
)
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Tablo 2b: BATADAL", styles["h3"]))
story.append(
    build_table(
        ["Model", "Orijinal (F1)", "Gürültülü (F1)", "Det. Rate", "Map. Acc."],
        [
            ["LSTM", "0.7642", "0.7624", "—", "—"],
            ["GRU", "0.7680", "0.7576", "—", "—"],
            ["1D-CNN", "0.7699", "0.7738", "—", "—"],
            ["Automata", "0.0202", "0.0582", "0.0012", "0.7500"],
        ],
        col_widths=[3.2 * cm, 3.2 * cm, 3.2 * cm, 2.8 * cm, 2.8 * cm],
    )
)
story.append(
    Paragraph(
        "<b>Unseen senaryo F1 (Automata):</b> SKAB = 0.0311 | BATADAL = 0.0202",
        styles["note"],
    )
)

# Section 3
story.append(Paragraph("3. Çapraz Veri Seti (Cross-Dataset) Genellenebilirliği", styles["h2"]))
story.append(Paragraph("Proje isterlerinde cross-dataset deneyi uygulanmamıştır.", styles["body"]))
story.append(Paragraph("Tablo 3: Cross-Dataset Performans Karşılaştırması", styles["h3"]))
story.append(
    build_table(
        ["Train / Test", "SKAB", "BATADAL"],
        [
            ["Train: SKAB", "—", "Uygulanmadı"],
            ["Train: BATADAL", "Uygulanmadı", "—"],
        ],
        col_widths=[5 * cm, 5 * cm, 5 * cm],
    )
)

# Section 4
story.append(Paragraph("4. Automata Parametre Duyarlılık Analizi (F1-score)", styles["h2"]))
story.append(Paragraph("Tablo 4a: SKAB", styles["h3"]))
story.append(
    build_table(
        ["Parametre", "Değer = 3", "Değer = 4", "Değer = 5", "Değer = 6"],
        [
            ["Window Size", "0.0141", "0.0140", "0.0115", "0.0440"],
            ["Alphabet Size", "0.0140", "0.0258", "0.0649", "0.0731"],
        ],
        col_widths=[4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm],
    )
)
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Tablo 4b: BATADAL", styles["h3"]))
story.append(
    build_table(
        ["Parametre", "Değer = 3", "Değer = 4", "Değer = 5", "Değer = 6"],
        [
            ["Window Size", "0.0000", "0.0202", "0.0325", "0.0274"],
            ["Alphabet Size", "0.0202", "0.1000", "0.0942", "0.1324"],
        ],
        col_widths=[4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm],
    )
)

# Section 5
story.append(Paragraph("5. Modellerin Çalışma Süresi (Runtime) Karşılaştırması", styles["h2"]))
story.append(
    Paragraph("Tüm koşuların ortalaması (logs/runtime_summary.csv).", styles["body"])
)
story.append(Paragraph("Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması", styles["h3"]))
story.append(
    build_table(
        ["Model", "Training Time (sn)", "Inference Time (sn)"],
        [
            ["LSTM", "55.04", "0.33"],
            ["GRU", "70.10", "0.30"],
            ["1D-CNN", "24.28", "0.18"],
            ["Automata", "0.01", "0.06"],
        ],
        col_widths=[5 * cm, 5 * cm, 5 * cm],
    )
)

# Summary
story.append(Paragraph("Özet Yorum", styles["h2"]))
story.append(
    build_table(
        ["Bulgu", "Değer"],
        [
            ["BATADAL en iyi DL F1", "1D-CNN: 0.7699 ± 0.0016"],
            ["SKAB en iyi DL F1", "1D-CNN: 0.2614 ± 0.0291"],
            ["En hızlı eğitim", "1D-CNN (24.28 sn)"],
            ["En yavaş eğitim", "GRU (70.10 sn)"],
            ["Gürültüden en çok faydalanan", "SKAB 1D-CNN (+0.0729 F1)"],
            ["Otomata en iyi parametre (BATADAL)", "Alphabet Size = 6 → F1 = 0.1324"],
            ["DL vs Automata (BATADAL, McNemar)", "Tüm seed'lerde p < 0.05"],
        ],
        col_widths=[7.5 * cm, 7.5 * cm],
    )
)

doc = SimpleDocTemplate(
    str(OUTPUT_PDF),
    pagesize=A4,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
    title="24_yazlab2_rapor",
    author="24. Grup - İbrahim Alperen KESKİN, Oğuzhan ATILKAN",
)

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(regular_font, 9)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Sayfa {doc.page}")
    canvas.drawString(2 * cm, 1.2 * cm, "YazLab 2 — 24. Grup Rapor Eki")
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"PDF oluşturuldu: {OUTPUT_PDF}")
