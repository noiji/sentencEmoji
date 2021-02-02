import logging

import azure.functions as func
from modules.my_translator import detect_and_translate


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        logging.info(f'Request body: {req_body}')
    except:
        return func.HttpResponse('Failed to load json request')
    
    try:
        text = req_body.get('text')

        return func.HttpResponse(detect_and_translate(text))

    except:
        return func.HttpResponse('Json in request was not in the correct format')