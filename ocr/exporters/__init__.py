"""Output exporter registry."""
from . import docx, json, odt, pdf, png, text

EXPORTERS = {
    "docx": docx.export,
    "json": json.export,
    "odt": odt.export,
    "pdf": pdf.export,
    "png": png.export,
    "txt": text.export,
}
