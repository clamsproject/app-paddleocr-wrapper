import argparse
import logging

import numpy as np
from clams import ClamsApp, Restifier
from lapps.discriminators import Uri
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from paddleocr import PaddleOCR


class PaddleocrWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        self.ocr = None

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug("running app")
        language = parameters['lang']
        self.ocr = PaddleOCR(lang=language)

        # process the text documents in the documents list
        for video_doc in mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            new_view.new_contain(AnnotationTypes.BoundingBox, document=video_doc.id)
            new_view.new_contain(DocumentTypes.TextDocument, document=video_doc.id)
            new_view.new_contain(Uri.SENTENCE, document=video_doc.id)
            new_view.new_contain(AnnotationTypes.Alignment)

            for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
                if 'label' in timeframe:
                    self.logger.debug(f'Found a time frame "{timeframe.id}" of label: "{timeframe.get("label")}"')
                else:
                    self.logger.debug(f'Found a time frame "{timeframe.id}" without label')
                if parameters.get("tfLabel") and \
                        'label' in timeframe and timeframe.get("label") not in parameters.get("tfLabel"):
                    continue
                else:
                    self.logger.debug(f'Processing time frame "{timeframe.id}"')
                if not timeframe.get("representatives"):
                    target_image: np.ndarray = vdh.extract_mid_frame(mmif, timeframe, as_PIL=False)
                    self.logger.debug("Extracted image")
                    self.logger.debug("Running OCR")
                    result = self.ocr.ocr(target_image)
                    text_content = ""
                    for layer1 in result:
                        if layer1 is not None:
                            source_id = representative.long_id
                            for layer2 in layer1:
                                bbox_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                                bbox_annotation.add_property("coordinates", layer2[0])
                                bbox_annotation.add_property("label", "text")
                                new_view.new_annotation(AnnotationTypes.Alignment, source=source_id, target=bbox_annotation.long_id)
                                sent_annotation = new_view.new_annotation(Uri.SENTENCE)
                                sent_annotation.add_property("text", layer2[1][0])
                                new_view.new_annotation(AnnotationTypes.Alignment, source=sent_annotation.long_id, target=bbox_annotation.long_id)
                                if text_content:
                                    text_content += "\n"
                                text_content += layer2[1][0]
                        text_document: Document = new_view.new_textdocument(text_content, text_content, language)
                        new_view.new_annotation(AnnotationTypes.Alignment, source=source_id, target=text_document.long_id)

                for rep_id in timeframe.get("representatives"):
                    representative: AnnotationTypes.TimePoint = input_view.get_annotation_by_id(rep_id)
                    rep_frame = vdh.convert(representative.get("timePoint"), "milliseconds",
                                    "frame", vdh.get_framerate(video_doc))
                    target_image: np.ndarray = vdh.extract_frames_as_images(video_doc, [rep_frame], as_PIL=False)[0]
                    self.logger.debug("Extracted image")
                    self.logger.debug("Running OCR")
                    result = self.ocr.ocr(target_image)
                    text_content = ""
                    for layer1 in result:
                        if layer1 is not None:
                            if representative.parent != new_view.id:
                                source_id = representative.long_id
                            else:
                                source_id = representative.id
                            for layer2 in layer1:
                                bbox_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                                bbox_annotation.add_property("coordinates", layer2[0])
                                bbox_annotation.add_property("boxType", "text")
                                new_view.new_annotation(AnnotationTypes.Alignment, source=source_id, target=bbox_annotation.id)
                                sent_annotation = new_view.new_annotation(Uri.SENTENCE)
                                sent_annotation.add_property("text", layer2[1][0])
                                new_view.new_annotation(AnnotationTypes.Alignment, source=sent_annotation.long_id, target=bbox_annotation.id)
                                if text_content:
                                    text_content += "\n"
                                text_content += layer2[1][0]
                            text_document: Document = new_view.new_textdocument(text_content)
                            new_view.new_annotation(AnnotationTypes.Alignment, source=source_id, target=text_document.long_id)
                            text_document.add_property("text", {"@language": language})
                            text_document.add_property("text", {"@value": text_content})
                
        if mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            image_doc: Document = mmif.get_documents_by_type(DocumentTypes.ImageDocument)[0]
            new_view: View = mmif.new_view()
            self.sign_view(new_view, parameters)
            new_view.new_contain(DocumentTypes.TextDocument)
            new_view.new_contain(AnnotationTypes.BoundingBox)
            new_view.new_contain(Uri.SENTENCE)
            new_view.new_contain(AnnotationTypes.Alignment)

            # run ocr annotation
            file_path = image_doc.location[7:] if image_doc.location.startswith('file://') else image_doc.location
            result = self.ocr.ocr(file_path)
            text_content = ""
            for layer1 in result:
                for layer2 in layer1:
                    bbox_annotation = new_view.new_annotation(AnnotationTypes.BoundingBox)
                    bbox_annotation.add_property("coordinates", layer2[0])
                    bbox_annotation.add_property("boxType", "text")
                    new_view.new_annotation(AnnotationTypes.Alignment, source=image_doc.long_id, target=bbox_annotation.long_id)
                    sent_annotation = new_view.new_annotation(Uri.SENTENCE)
                    sent_annotation.add_property("text", layer2[1][0])
                    new_view.new_annotation(AnnotationTypes.Alignment, source=sent_annotation.long_id, target=bbox_annotation.long_id)
                    if text_content:
                        text_content += "\n"
                    text_content += layer2[1][0]
            text_document: Document = new_view.new_textdocument(text_content)
            new_view.new_annotation(AnnotationTypes.Alignment, source=image_doc.long_id, target=text_document.long_id)
            text_document.add_property("text", {"@language": language})
            text_document.add_property("text", {"@value": text_content})

        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = PaddleocrWrapper()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
