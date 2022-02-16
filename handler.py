from abc import ABC
import json
import logging
import os

import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ts.torch_handler.base_handler import BaseHandler

from transformers import ElectraModel, ElectraTokenizer

logger = logging.getLogger(__name__)

from torch import nn

class ELECTRAClassifier(nn.Module): #0216 추가
    def __init__(self,
                 electra,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):

        super(ELECTRAClassifier, self).__init__()

        self.electra = electra

        # # do not train electra parameters
        for p in self.electra.parameters():
            p.requires_grad = False

        self.dr_rate = dr_rate

        # self.classifier = nn.Linear(hidden_size , num_classes)

        #         # 방법 1 -> forward에서 처리해줘야 함.
        #         self.classifier1 = nn.Linear(hidden_size, 100) # y = Wx
        #         self.classifier2 = nn.Linear(100, num_classes) # z = Uy
        #         #layer 추가 시 activation function을 주지 않으면 의미가 없음.
        #         self.relu = nn.ReLU()

        # 방법 2 -> forward에서 별도 처리 필요 X
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 100), nn.ReLU(), nn.Linear(100, num_classes))

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, token_type_ids, attention_mask):

        # # eval: drop out 중지, batch norm 고정과 같이 evaluation으로 모델 변경
        # self.electra.eval()

        # gradient 계산을 중지
        with torch.no_grad():
            # ElectraModel은 pooled_output을 리턴하지 않는 것을 제외하고 BertModel과 유사합니다.
            #            x = self.electra(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output

            x = self.electra(input_ids=input_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask).last_hidden_state[:, 0, :]
            # .last_hidden_state[:, 0, :]: [batch , CLS 위치, depth]

            # Sentence Embedding으로 무엇을 넣을까? CLS, average, ... (Bert의 경우에는 Sentence BERT라는 게 제안되었다고 함)

        x = self.dropout(x)

        return self.classifier(x)

class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        logger.debug("Initializing...:)")
        # self.manifest = ctx.manifest

        # properties = ctx.system_properties

        # model_dir = properties.get("model_dir")

        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        model_dir = "C:/Users/enoch9/Desktop/개발/sentencEmoji/"

        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
        self.model = ElectraModel.from_pretrained(model_dir)
        logger.info("checkpoint model found.")

        self.model.to(self.device)
        self.model.eval()

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        logger.debug("Preprocessing...:)")

        """ Very basic preprocessing code - only tokenizes.
            Extend with your own preprocessing steps as needed.
        """

        #추후 여기서 sentence split 필요.( or @backend server) json으로 불러오는 것 같은데 내 모델에도 동일할지?
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        # inputs = self.tokenizer.encode_plus(
        #     sentences,
        #     add_special_tokens=True,
        #     return_tensors="pt"
        # )

        inputs = self.tokenizer(
            str(sentences),
            return_tensors='pt')

        return inputs

    def inference(self, inputs):
        logger.debug("Inferencing...:)")
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit
        # its expected input format.
        prediction = self.model(
            inputs['input_ids'].to(self.device),
            token_type_ids=inputs['token_type_ids'].to(self.device)
        )[0].argmax().item()
        logger.info("(Before) Model predicted: '%s'", prediction)

        ####
        #My Model: model(input_ids, token_type_ids, attention_mask)
        #Attention mask 넣든 안넣든 같은 값을 출력함
        prediction = self.model(
            inputs['input_ids'].to(self.device),
            token_type_ids=inputs['token_type_ids'].to(self.device),
            attention_mask = inputs['attention_mask'].to(self.device),
        )[0].argmax().item()

        #model()의 결과:
        #tensor([[-4.3821,  4.9184,  7.5785, -3.6893, -9.4861],
        #        [-3.5774,  4.2812,  6.3018, -2.7638, -8.4373],
        #        [-4.5047,  4.9979,  7.2914, -3.5938, -9.3536],
        #         ...
        # 6957
        #model.py에서 out[0].argmax().item() 하면 0~4 나오는데..

        logger.info("(After) Model predicted: '%s'", prediction)
        #####

        if self.mapping:
            prediction = self.mapping[str(prediction)]

        return [prediction]

    def postprocess(self, inference_output):
        logger.debug("Postprocessing...:)")
        #여기서 emoji class -> emoji random select 필요
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    logger.debug("Handling...:)")
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
