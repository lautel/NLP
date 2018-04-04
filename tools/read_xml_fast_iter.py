import argparse, cPickle, codecs
from lxml import etree
from tag2vec import tag2embed

class readXML():
    def __init__(self):
        self.tag_dict = {}
        self.tag_vocabulary = []
        self.counter = 0

    def _set_up_context(self, xmlfile):
        return etree.iterparse(xmlfile, tag='token', events=('end', ))

    def get_results(self):
        return self.tag_dict, self.tag_vocabulary

    # In order to free some memory as you parse:
    def fast_iter(self, xmlfile, attribute, valid_dict):
        """
        Help found at:
        https://stackoverflow.com/questions/12160418/why-is-lxml-etree-iterparse-eating-up-all-my-memory
        http://lxml.de/parsing.html#modifying-the-tree
        Based on Liza Daly's fast_iter:
        http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
        """
        context = self._set_up_context(xmlfile)

        for _, elem in context:
            self._parse_xml(elem, attribute, valid_dict)
            # It's safe to call clear() here because no descendants will be accessed
            elem.clear()
            # Also eliminate now-empty references from the root node to elem
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]
        del context
        print("%d different pos tag found" % len(self.tag_vocabulary))
        print("%d words with more than one syntactic value" % self.counter)

    def _parse_xml(self, elem, attribute, valid_dict):
        if elem.attrib.get('lemma') in valid_dict:
            value = elem.attrib.get(attribute)
            if value not in self.tag_vocabulary:
                self.tag_vocabulary.append(value)
            if elem.attrib['lemma'] in self.tag_dict:  # Word already stored in dictionary
                if elem.attrib.get(attribute) == value:  # Same POS tag
                    return
                else:
                    self.tag_dict[elem.attrib['lemma']].append(value)  # Add new syntactic information
                    self.counter += 1
            else:
                self.tag_dict[elem.attrib['lemma']] = [value]
        else:
            pass


def open_vocabulary(filepath):
    vocab = codecs.open(filepath,'r','utf-8')
    vocab = list(vocab)

    words = [x.rstrip().split()[0] for x in vocab]

    w2idx = {w: idx for idx, w in enumerate(words)}
    idx2w = {idx: w for idx, w in enumerate(words)}

    return w2idx, idx2w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-xml', help='XML file', dest='xml', default='/extra/scratch03/Language/Periodicos/textos/2.docsent2text/upto201709.text.xml')
    parser.add_argument('-out', help='Path to the output variable', dest='out', required=False,
                        default='embeddings/upto201709.text.postag_dictionary.pkl')
    parser.add_argument('-v', help='Vocabulary after filtering out words with count-min threshold. It is used in tag2vec to compute the embeddings',
                        dest='vocab', default='../../WordRepresentation/Word2vec/upto201709.text.lema.200.wordcount.txt')
    parser.add_argument('-attrib', help='Attribute extracted', dest='atr', default='tag')
    args = parser.parse_args()

    try:
        postag_dict = cPickle.load(open(args.out,"r"))
        dictionary = True
    except IOError:
        dictionary = False

    if not dictionary:
        voc, _ = open_vocabulary(args.vocab)
        print("Ready to parse the XML file")
        # Parse the XML file and save the vocabulary and its POS tags
        reader = readXML()
        reader.fast_iter(args.xml, args.atr, voc)
        postag_dict, postag_vocab = reader.get_results()
        cPickle.dump(postag_dict, open(args.out,"wb"))
        tag2embed(postag_dict, args.vocab)
    else:
        print("Dictionary with POS tags loaded")
        # Dictionary with vocabulary and POS tags already created
        tag2embed(postag_dict, args.vocab)
