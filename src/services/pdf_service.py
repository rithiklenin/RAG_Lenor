import os
import json
import uuid
import logging
import io
import zipfile
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams

class PDFExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.unique_id = str(uuid.uuid4())
        self.extracted_data = None
        self._extract_pdf()

    def _extract_pdf(self):
        try:
            with open(self.file_path, "rb") as file:
                input_stream = file.read()

            credentials = ServicePrincipalCredentials(
                client_id=os.getenv("ADOBE_SERVICES_CLIENT_ID"),
                client_secret=os.getenv("ADOBE_SERVICES_CLIENT_SECRET")
            )

            pdf_services = PDFServices(credentials=credentials)
            input_asset = pdf_services.upload(
                input_stream=input_stream,
                mime_type=PDFServicesMediaType.PDF
            )

            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT]
            )

            extract_pdf_job = ExtractPDFJob(
                input_asset=input_asset,
                extract_pdf_params=extract_pdf_params
            )

            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
            result_asset = pdf_services_response.get_result().get_resource()
            stream_asset = pdf_services.get_content(result_asset)

            zip_bytes = io.BytesIO(stream_asset.get_input_stream())
            with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
                self.extracted_data = {
                    name: zip_ref.read(name) for name in zip_ref.namelist()
                }

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f"Exception encountered while executing operation: {e}")

    def get_extracted_data(self):
        if "structuredData.json" in self.extracted_data:
            return json.loads(self.extracted_data["structuredData.json"])
        return None

def get_text_chunks(filename, json_data):
    """Extract text chunks from PDF data."""
    if "elements" not in json_data:
        logging.error("Missing 'elements' key in json_data")
        raise ValueError("Missing 'elements' key in json_data")

    page_text = ""
    start_page = 0
    all_texts = []
    list_label = ""

    for i, element in enumerate(json_data["elements"]):
        try:
            current_page = element["Page"]
            
            if current_page > start_page:
                if page_text:
                    all_texts.append({
                        "ElementType": "Text",
                        "file_name": filename,
                        "Page": start_page,
                        "Text": page_text
                    })
                start_page = current_page
                page_text = ""
                list_label = ""
            
            if "Text" in element:
                if element["Path"].endswith("Lbl") and not element["Path"].startswith("//Document/Table"):
                    list_label = element["Text"]
                else:
                    if list_label:
                        page_text += f"{list_label} {element['Text']}\n"
                        list_label = ""
                    else:
                        page_text += f"{element['Text']}\n"
                        
        except KeyError as e:
            logging.warning(f"Key error in json_data['elements'][{i}]: {e}")

    # Add the last page
    if page_text:
        all_texts.append({
            "ElementType": "Text",
            "file_name": filename,
            "Page": current_page,
            "Text": page_text
        })

    return all_texts 