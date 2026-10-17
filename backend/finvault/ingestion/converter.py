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
IMAGE_PLACEHOLDER = "<!--IMG-->"


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
                # Pictures covering less than this fraction of the page are
                # silently skipped (no VLM call, no error); the default 0.05
                # drops the ~4.8%-of-page photo in tate.pdf.
                picture_area_threshold=0.01,
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
        result = self._converter.convert(pdf_path)
        document = result.document
        markdown: str = document.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER,  # No image, just placeholder
            image_placeholder=IMAGE_PLACEHOLDER,
            page_break_placeholder="<!-- page_break -->",
            # The meta serializer would render picture.meta.description as
            # plain text; we inject our own tag below instead.
            blocked_meta_names={"description"},
        )
        # docling 2.70 stores descriptions in picture.meta, which disables the
        # legacy annotation markers in the markdown export, so inject the tags
        # ourselves, replacing each placeholder in document order.
        for picture in document.pictures:
            text = (
                picture.meta.description.text
                if picture.meta and picture.meta.description
                else None
            )
            tag = f"<image_description>\n{text}\n</image_description>" if text else ""
            markdown = markdown.replace(IMAGE_PLACEHOLDER, tag, 1)
        return markdown
