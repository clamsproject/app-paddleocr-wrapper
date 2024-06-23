"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata

from lapps.discriminators import Uri

# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    

    # first set up some basic information
    metadata = AppMetadata(
        name="paddleocr wrapper",
        description='CLAMS app wraps the PaddleOCR model (https://github.com/PaddlePaddle/PaddleOCR). The '
                    'model can detect text regions in the input image and recognize text in the regions. '
                    'The model support multiple languages (80+), including chinese and English. For details of supported languages, go to PaddleOCR webpage.'
                    'The model also support both video and image as input. When using this app for ocr in video, the input mmif must come with input view, which can be generized by using other CLAMS APP.', 
        app_license="Apache 2.0",  
        identifier="paddleocr-wrapper",  
        url="https://github.com/clamsproject/app-paddleocr-wrapper", 
        analyzer_version='2.7.3',
        analyzer_license="Apache 2.0",
    )
    # and then add I/O specifications: an app must have at least one input and one output
    metadata.add_input(DocumentTypes.ImageDocument)
    metadata.add_input(DocumentTypes.VideoDocument)
    in_tf = metadata.add_input(AnnotationTypes.TimeFrame, representatives='?', required=False)
    in_tf.add_description('The Time frame annotation that represents the video segment to be processed. When '
                          '`representatives` property is present, the app will process videos still frames at the '
                          'underlying time point annotations that are referred to by the `representatives` property. '
                          'Otherwise, the app will process the middle frame of the video segment.')
    out_sent = metadata.add_output(at_type=Uri.SENTENCE, text='*')
    out_sent.add_description('The smallest recognized unit of PaddleOCR "lines" in the input images. `text` property stores '
                             'the string value of recognized words.')
    out_td = metadata.add_output(DocumentTypes.TextDocument)
    out_td.add_description('Fully serialized text content of the recognized text in the input images. Serialization is'
                           'done by concatenating `text` values of `Sentence` annotations with two newline characters.')
    out_bbox = metadata.add_output(AnnotationTypes.BoundingBox)
    out_bbox.add_description('Bounding boxes of the detected text regions in the input images. No corresponding box '
                             'for the entire image (`TextDocument`) region')
    out_ali = metadata.add_output(AnnotationTypes.Alignment)
    out_ali.add_description('Alignments between 1) `TimePoint` <-> `TextDocument`, 2) `BoundingBox` <-> `Sentence`, 3) `TimePoint` <-> `BoundingBox`')
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(name='lang', default="en", 
                           description='The target language of OCR',
                           type='string')
    
    metadata.add_parameter(name='tfLabel', default=[], type='string', multivalued=True,
                           description='The label of the TimeFrame annotation to be processed. By default (`[]`), all '
                                       'TimeFrame annotations will be processed, regardless of their `label` property '
                                       'values.')
    # metadta.add_parameter(more...)
    
    # CHANGE this line and make sure return the compiled `metadata` instance
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
