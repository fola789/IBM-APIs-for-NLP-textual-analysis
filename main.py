import abc
import sys
import json
import click

from watson_developer_cloud import LanguageTranslatorV3
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features,\
    EntitiesOptions, SentimentOptions


class TextOperation(abc.ABC):

    @abc.abstractmethod
    def produce_output(self, input_text):
        pass


class TextTranslator(TextOperation):

    def __init__(self, dst_lang, api_key):
        self.__dst_lang = dst_lang

        api_url = 'https://gateway-lon.watsonplatform.net/' \
                  'language-translator/api'
        self.__translator = LanguageTranslatorV3(version='2019-02-28',
                                                 url=api_url,
                                                 iam_apikey=api_key)

    def produce_output(self, input_text):
        translation = self.__translator.translate(
            text=input_text, model_id='en-{}'.format(self.__dst_lang)).\
            get_result()
        return translation['translations'][0]['translation']


class NluProcessor(TextOperation, abc.ABC):

    def __init__(self, api_key):
        api_url = 'https://gateway-lon.watsonplatform.net/' \
                  'natural-language-understanding/api'
        self._nlu_processor = NaturalLanguageUnderstandingV1(
            version='2019-02-28',
            url=api_url,
            iam_apikey=api_key)


class SentimentAnalyzer(NluProcessor):

    def __init__(self, api_key):
        super().__init__(api_key)

    def produce_output(self, input_text):
        response = self._nlu_processor.analyze(
            text=input_text,
            features=Features(sentiment=SentimentOptions())).get_result()
        score = response['sentiment']['document']['score']
        label = response['sentiment']['document']['label']

        return 'score: {:.5f} | label: {}'.format(score, label)


class EntityExtractor(NluProcessor):

    def __init__(self, api_key):
        super().__init__(api_key)

    def produce_output(self, input_text):
        response = self._nlu_processor.analyze(
            text=input_text,
            features=Features(entities=EntitiesOptions())).get_result()

        entities = response['entities']
        return json.dumps(entities, indent=2)


@click.command()
@click.argument('operation_mode')
@click.option('--oper-param', default=None,
              help='additional parameter for the operation')
def main(operation_mode, oper_param):
    operation_inst = None

    if operation_mode == 'translate':
        api_key = '6_-d36HjV1lqn7KFNk6ADdreE2_wBx1W01ipUAbSlG7U'
        operation_inst = TextTranslator(oper_param, api_key)
    elif operation_mode == 'sentiment_analysis':
        api_key = 'tqa-ZxvMqKZwIcyrX6qtmrzcSWWT97HZ7tAXFnlMVu8R'
        operation_inst = SentimentAnalyzer(api_key)
    elif operation_mode == 'entity_extraction':
        api_key = 'tqa-ZxvMqKZwIcyrX6qtmrzcSWWT97HZ7tAXFnlMVu8R'
        operation_inst = EntityExtractor(api_key)
    else:
        raise ValueError('error: unsupported operation mode: {}'.
                         format(operation_mode))

    print(operation_inst.produce_output(sys.stdin.read()))

    return 0


if __name__ == '__main__':
    sys.exit(main())
