from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter as DoclingConverter
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import ImageRefMode

from finvault.prompts import VLM_IMAGE_DESCRIPTION, load_prompt

OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "qwen3-vl:2b"


class DocumentConverter:
    """Converts PDF documents to markdown with table extraction and image descriptions."""

    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            enable_remote_services=True,
            do_ocr=False,
            do_table_structure=True,
            generate_picture_images=True,
            do_picture_description=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE
            ),
            picture_description_options=PictureDescriptionApiOptions(
                url=f"{OLLAMA_URL}/v1/chat/completions",
                params={
                    "model": VLM_MODEL,
                    "think": False,
                    "seed": 42,
                    "max_completion_tokens": 256,
                },
                prompt=load_prompt(VLM_IMAGE_DESCRIPTION),
                timeout=90,
            ),
        )
        self._converter = DoclingConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    def convert(self, pdf_path: str) -> str:
        """Convert PDF to markdown with table extraction and image descriptions."""
        # TODO: Use the Docling converter to process the PDF file.
        # TODO: Transform annotation markers into semantic image description tags.
        pass
