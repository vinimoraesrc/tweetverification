import re

FLAGS = re.MULTILINE | re.DOTALL


def hashtag_glove(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "{}".format(hashtag_body.lower())
    else:
        split = []
        result = " ".join(
            ["<hashtag>"] + [hashtag_body])
    return result


def allcaps_glove(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def preprocess_glove_ruby_port(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes,
                                                  nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes,
                                            nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag_glove)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    text = re_sub(r"([A-Z]){2,}", allcaps_glove)

    return text.lower()


def preprocess_glove_ruby_port_list(text_list):
    prepro = [preprocess_glove_ruby_port(text) for text in text_list]
    return [p for p in prepro if len(p) > 0]


def preprocess_glove_ruby_port_authors(authors):
    return [preprocess_glove_ruby_port_list(text_list) for text_list in authors]
