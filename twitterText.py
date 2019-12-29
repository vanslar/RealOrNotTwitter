from nltk.tokenize import TweetTokenizer
import re

tok = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles = True)

class TextPreprocesser:
    def __init__(self, tokenizer = tok):
        self.tok = tokenizer

    def preProcess(self, text):
        s = ' '.join(self.tok.tokenize(text))
        s = self._removeHttpText(s)
        return s

    def _removeHttpText(self, text):
        pattern = r'http\S+'
        text = re.sub(pattern, ' ', text)
        return text

#text_pres = TextPreprocesser(tok)
#print(text_pres.preProcess(train_fd['text'][0]))