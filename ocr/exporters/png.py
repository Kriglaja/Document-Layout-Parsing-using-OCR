"""Project visualization PNG exporter."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..models import OcrToken, ProcessedDocument

COLORS = ["#3969AC", "#11A579", "#E73F74", "#F2B701", "#7F3C8D", "#80BA5A", "#E68310", "#008695"]


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def word_mask(token: OcrToken) -> Image.Image:
    width = max(1, int(token.bbox.width))
    height = max(1, int(token.bbox.height))
    face = font(max(16, height * 3))
    probe = Image.new("L", (max(256, width * 6), max(128, height * 6)), 0)
    draw = ImageDraw.Draw(probe)
    bounds = draw.textbbox((0, 0), token.text, font=face)
    ink = Image.new("L", (max(1, bounds[2] - bounds[0]), max(1, bounds[3] - bounds[1])), 0)
    ImageDraw.Draw(ink).text((-bounds[0], -bounds[1]), token.text, fill=255, font=face)
    return ink.resize((width, height), Image.Resampling.LANCZOS)


def paste_token(canvas: Image.Image, token: OcrToken) -> None:
    mask = word_mask(token)
    position = (int(token.bbox.x), int(token.bbox.y))
    canvas.paste(Image.new("RGB", mask.size, "black"), position, mask)


def export(document: ProcessedDocument, output_directory: Path) -> None:
    page_root = output_directory / f"{document.name}-pages"
    page_root.mkdir(parents=True, exist_ok=True)

    for page in document.pages:
        source = Image.open(page.image_path).convert("RGB")
        layout = source.copy()
        layout_draw = ImageDraw.Draw(layout)
        spatial = Image.new("RGB", source.size, "white")
        token_boxes = Image.new("RGB", source.size, "white")
        token_draw = ImageDraw.Draw(token_boxes)
        hybrid = Image.new("RGB", source.size, "white")

        labels = sorted({region.label for region in page.regions})
        colors = {label: COLORS[index % len(COLORS)] for index, label in enumerate(labels)}

        for region in page.regions:
            x = int(region.bbox.x)
            y = int(region.bbox.y)
            width = int(region.bbox.width)
            height = int(region.bbox.height)
            rectangle = (x, y, x + width, y + height)
            color = colors[region.label]
            layout_draw.rectangle(rectangle, outline=color, width=3)
            layout_draw.text(
                (x, max(0, y - 15)),
                f"{region.label} {region.score:.2f}",
                fill=color,
                font=font(13, bold=True),
            )

            if region.action == "visual":
                hybrid.paste(source.crop(rectangle), (x, y))

            for token in region.tokens:
                paste_token(spatial, token)
                paste_token(token_boxes, token)
                paste_token(hybrid, token)
                token_rectangle = (
                    int(token.bbox.x),
                    int(token.bbox.y),
                    int(token.bbox.x + token.bbox.width),
                    int(token.bbox.y + token.bbox.height),
                )
                token_draw.rectangle(token_rectangle, outline="#D8A600", width=1)

        page_directory = page_root / page.image_key
        page_directory.mkdir(parents=True, exist_ok=True)
        layout.save(page_directory / "01-layout.png")
        spatial.save(page_directory / "02-ocr-spatial.png")
        token_boxes.save(page_directory / "03-ocr-token-boxes.png")
        hybrid.save(page_directory / "04-ocr-hybrid.png")

        gap = 20
        heading_height = 45
        comparison = Image.new(
            "RGB",
            (source.width * 2 + gap, source.height + heading_height),
            "white",
        )
        comparison_draw = ImageDraw.Draw(comparison)
        comparison_draw.text((10, 8), "Detected layout", fill="black", font=font(20, bold=True))
        comparison_draw.text(
            (source.width + gap + 10, 8),
            "Spatial OCR",
            fill="black",
            font=font(20, bold=True),
        )
        comparison.paste(layout, (0, heading_height))
        comparison.paste(token_boxes, (source.width + gap, heading_height))
        comparison.save(page_directory / "05-comparison.png")
