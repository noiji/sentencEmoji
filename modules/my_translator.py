from google_trans_new import google_translator  
import http.client, urllib.parse, json, time, sys


def detect_and_translate(text):
    translator = google_translator()  

    #language detection
    lang = translator.detect(text) 
    org_lang = lang[0] #later will be used for answer translation

    if org_lang != 'en':
        text = translator.translate(text, lang_tgt = 'en')

    host = "swfc2021.azurewebsites.net"
    endpoint_key = "dbbaa754-48f2-4303-8094-5ba7484cd68d"
    route = "/qnamaker/knowledgebases/ac4613b0-c61e-4fd7-a4b6-b55caf548834/generateAnswer"

    questions = {'question': text,'top': 3}
    questions = str(questions)

    headers = {
        'Authorization': 'EndpointKey ' + endpoint_key,
        'Content-Type': 'application/json'
    }


    try:
        conn = http.client.HTTPSConnection(host,port=443)
        conn.request ("POST", route,  questions, headers)
        response = conn.getresponse ()
        answer = response.read ()

        ret = json.loads(answer)
        answer = ret['answers'][0]['answer']
        return answer

    except :
        answer = "Sorry. Can't find an answer. (HTTPS Connection Error :<)"
        return answer
    
    if org_lang != 'en':
        answer = translator.translate(answer, lang_tgt = org_lang)  
    
    return answer
